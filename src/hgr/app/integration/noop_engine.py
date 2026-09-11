from __future__ import annotations

import math
import os
import queue
import re
import sys
import threading
import time
import traceback
from collections import deque
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Optional

import cv2
import numpy as np
from PySide6.QtCore import Qt, QObject, QTimer, Signal

from ...debug.chrome_controller import ChromeController
from ...debug.chrome_gesture_router import ChromeGestureRouter
from ...debug.foreground_window import get_foreground_window_info, is_foreground_fullscreen
from ...debug.mouse_controller import MouseController
from ...debug.mouse_gesture import MouseGestureTracker
from ...voice.live_dictation import LiveDictationEvent, LiveDictationStreamer
from ...config.app_config import save_config
from ...config.gesture_bindings import (
    STATIC_LABEL_TO_POSE,
    STATIC_POSE_LABEL_MAP,
    action_bound_to_pose,
    default_pose_for_action,
    pose_id_for_static_label,
    static_label_for_pose_id,
)
from ...debug.low_fps_suggestion_overlay import LowFpsSuggestionOverlay
from ...debug.screen_volume_overlay import ScreenVolumeOverlay
from ...debug.spotify_controller import SpotifyController
from ...debug.spotify_gesture_router import SpotifyGestureRouter
from ...debug.discord_controller import DiscordController
from ...debug.discord_gesture_router import DiscordGestureRouter, DiscordGestureSnapshot
from ...debug.text_input_controller import TextInputController
from ...debug.voice_command_listener import VoiceCommandListener
from ...debug.volume_controller import VolumeController
from ...debug.volume_gesture import VolumeGestureTracker
from ...debug.youtube_controller import YouTubeController
from ...debug.youtube_gesture_router import YouTubeGestureRouter
from ...gesture.recognition.engine import GestureRecognitionEngine
from ...gesture.tracking.detector import HandDetector
from ...gesture.tracking.smoothing import AdaptiveLandmarkSmoother, OneEuroFilter
from ...gesture.ui.test_window import SpotifyWheelOverlay
from ...gesture.ui.voice_status_overlay import VoiceStatusOverlay
from ...voice.command_processor import VoiceCommandContext, VoiceCommandProcessor
from ...voice.dictation import DictationProcessor
from ...voice.grammar_corrector import CorrectionResult, GrammarCorrector
from ...voice.llama_server import LlamaServer
from ...voice.whisper_refiner import RefinementResult, WhisperRefiner
# Single source of truth for whether streaming live-dictation hypotheses are on
# (env HGR_DICTATION_HYPOTHESES). Used to keep the refiner mutually exclusive
# with live typing — see _start_voice_capture.
from ...voice.whisper_stream import _HYP_ENABLED as _DICTATION_HYP_ENABLED
from ..camera.camera_utils import open_camera_by_index, open_phone_camera_url, open_preferred_or_first_available
from ..camera.ffmpeg_capture import FfmpegMjpegCapture, open_ffmpeg_cap_with_fps_fallback, resolve_dshow_device_for_index


_DICTATION_HALLUCINATION_STOPWORDS = {
    "the", "you", "and", "a", "to", "of", "is", "it", "so", "i",
    "uh", "um", "ah", "oh", "mm", "mhm", "hmm", "hm", "eh",
    "thanks", "thank", "bye", "okay", "ok",
}


# Voice commands recognized only when the committed utterance is EXACTLY one
# of these phrases (after punctuation strip + whitespace collapse + lowercase).
# The whole-utterance match is the safety â€” saying "I need a new line of code"
# in a sentence won't trigger because the commit contains more than just the
# command phrase. The user has to pause, say just the command, then pause.
_DICTATION_NEWLINE_COMMANDS = frozenset({
    "new line", "newline", "next line", "line break",
    "press enter", "press return", "hit enter", "enter key",
})
_DICTATION_PARAGRAPH_COMMANDS = frozenset({
    "new paragraph", "paragraph break",
})
_DICTATION_COMMAND_NORMALIZE_RE = re.compile(r"[^\w\s]+")


def _parse_dictation_command(text: str) -> Optional[str]:
    if not text:
        return None
    stripped = _DICTATION_COMMAND_NORMALIZE_RE.sub("", text)
    normalized = re.sub(r"\s+", " ", stripped).strip().lower()
    if not normalized:
        return None
    if normalized in _DICTATION_NEWLINE_COMMANDS:
        return "newline"
    if normalized in _DICTATION_PARAGRAPH_COMMANDS:
        return "paragraph"
    return None


# Left-hand "one" (voice listen) is a *held* pose, not a swipe. Net palm
# travel is measured in palm-scale units from the start of the hold.
# Landmark jitter is typically ~0.05–0.12; a small drift while holding
# is fine. A committed swipe is usually >0.6. This cap sits in between
# so the gate is not "perfectly still".
_VOICE_ONE_MAX_HOLD_TRAVEL_PALM = 0.40


def _palm_net_travel_exceeds(
    origin_xy: tuple[float, float],
    current_xy: tuple[float, float],
    palm_scale: float,
    *,
    max_travel: float = _VOICE_ONE_MAX_HOLD_TRAVEL_PALM,
) -> bool:
    scale = max(float(palm_scale), 1e-6)
    dx = (float(current_xy[0]) - float(origin_xy[0])) / scale
    dy = (float(current_xy[1]) - float(origin_xy[1])) / scale
    return math.hypot(dx, dy) > max_travel


_DICTATION_TRAILING_HALLUCINATIONS = {"the", "you", "and", "a"}

_WHISPER_STOCK_HALLUCINATIONS = (
    "good afternoon, everyone",
    "good afternoon everyone",
    "good morning, everyone",
    "good morning everyone",
    "thank you for watching",
    "thanks for watching",
    "please subscribe",
    "like and subscribe",
    "don't forget to subscribe",
    "bye-bye",
    "bye bye",
    "we'll be right back",
    "we will be right back",
    "see you in the next one",
    "see you next time",
)

_WHISPER_STOCK_PATTERNS = tuple(
    re.compile(r"\b" + re.escape(phrase) + r"\.?", re.IGNORECASE)
    for phrase in _WHISPER_STOCK_HALLUCINATIONS
) + (
    # Windows / MS Office ProgID hallucinations whisper pulls from training
    # data on low-signal trailing audio (e.g., mic bumps, keyboard noise).
    # The ProgID-like suffix ("MSWordDoc Word.Document.8" or similar) is the
    # distinguishing signal â€” legitimate phrases like "I opened a word
    # document" never include it, so the suffix is required.
    re.compile(
        r"\b(?:microsoft\s+)?(?:word|excel|powerpoint)\s+document"
        r"\s+ms(?:word|excel|powerpoint)doc[.\s]*"
        r"(?:word|excel|powerpoint)\.?document\.?\d*\.?",
        re.IGNORECASE,
    ),
)


_SENTENCE_SPLIT_RE = re.compile(r"[^.!?]+[.!?]+\s*|[^.!?]+$")


def _collapse_repeated_sentences(text: str) -> str:
    # Whisper-stream sometimes emits the same sentence 2-3x concatenated into
    # a single final (trailing-silence hallucination after a real utterance).
    parts = _SENTENCE_SPLIT_RE.findall(text)
    if len(parts) <= 1:
        return text
    out: list[str] = []
    last_norm = ""
    for part in parts:
        norm = re.sub(r"\s+", " ", part.strip(" .!?,;:\"'").lower())
        if norm and norm == last_norm:
            continue
        out.append(part)
        last_norm = norm
    return "".join(out).strip()


def _norm_token(tok: str) -> str:
    # Case/punctuation-insensitive word key used to reconcile the rough
    # live-typed dictation prefix against the high-accuracy final.
    return tok.lower().strip(".,!?;:\"'")


def _reconcile_final_edit(committed: str, committed_chars: int, final_text: str) -> tuple[int, str, int]:
    """Diff a dictation final against the live-typed `committed` prefix.

    Returns ``(backspace_chars, to_type, common_chars)``:
      * ``backspace_chars`` — chars to delete from the END of the on-screen live
        text to drop the divergent/over-long tail. Always within
        ``[0, committed_chars]`` so it can never reach past this utterance's own
        live text into prior committed finals or pre-dictation user text.
      * ``to_type`` — text to insert after the agreed prefix; includes the
        leading separator space (when a non-empty agreed prefix precedes it) and
        a trailing separator space.
      * ``common_chars`` — chars of the agreed prefix left untouched on screen.

    The comparison is by WORD, case/punctuation-insensitively: greedy live
    partials and the beam final frequently differ only in casing/punctuation, so
    a char-exact ``startswith`` would treat the whole final as new and re-type it
    on top of the live text (duplication). Keeping the agreed prefix and only
    re-typing the divergent tail prevents that. Commit-only callers pass
    ``committed=""`` / ``committed_chars=0`` and get back ``(0, final+" ", 0)`` —
    i.e. the entire final, exactly the pre-streaming behaviour.
    """
    cw = committed.split()
    fw = final_text.split()
    k = 0
    while k < len(cw) and k < len(fw) and _norm_token(cw[k]) == _norm_token(fw[k]):
        k += 1
    common_chars = len(" ".join(cw[:k]))  # agreed prefix, no trailing space
    backspace_chars = max(0, committed_chars - common_chars)
    tail = " ".join(fw[k:])
    if tail:
        to_type = ("" if k == 0 else " ") + tail + " "
    else:
        # final equals / is a prefix of the live text -> just terminate with a
        # separator space (any over-long live tail was backspaced above).
        to_type = " "
    return backspace_chars, to_type, common_chars


# Common words that should stay lowercase mid-sentence. whisper capitalizes the
# first word of EACH utterance, so when the user pauses mid-sentence the next
# utterance's first word comes back wrongly capitalized ("plenty of [pause] free"
# -> "Free"). We only ever de-capitalize words in this set, never acronyms or
# proper nouns, so "pushed it to [pause] GitHub" keeps "GitHub" and "[pause] API"
# keeps "API".
_SEAM_LOWERCASE_WORDS = frozenset({
    "a", "an", "and", "the", "but", "so", "or", "nor", "for", "yet",
    "we", "it", "they", "he", "she", "you", "that", "this", "these", "those",
    "of", "in", "on", "at", "to", "by", "with", "from", "as", "if", "when",
    "then", "than", "one", "two", "free", "need", "just", "also", "actually",
    "okay", "well", "is", "are", "was", "were", "be", "been", "do", "does",
    "did", "can", "could", "should", "would", "will", "not", "no", "yes",
    "my", "our", "your", "their", "its", "because", "while", "after", "before",
    "since", "until", "though", "although", "however", "maybe", "really",
    "even", "still", "about", "into", "over", "under", "out", "up", "down",
})

# Chars after which a new utterance does NOT need a leading separator space.
_SEAM_NO_LEAD_AFTER = (" ", "\n", "\t", "(", "[", "{", "“", '"', "-", "/")


def _seam_normalize(text: str, prev_display: str, last_char: str) -> tuple[str, str]:
    """Normalize the FIRST content of a new utterance where it joins the
    running transcript. Returns ``(lead_space, cased_text)``:

      * ``lead_space`` — "" or " ". A single separator space is needed only when
        there is prior text AND the previously-typed char isn't already a
        separator/opening bracket. In the normal case the prior final left a
        trailing space (last_char == " "), so this is "" and nothing doubles; it
        only fills in a space that some path dropped ("ofFree" -> "of free").
      * ``cased_text`` — ``text`` with its first letter re-cased: capitalized at a
        sentence start (prev ended with . ! ? : ; or newline, or there's no prior
        text), else lowercased IFF the first word is a common lowercase word
        (undoing whisper's per-utterance initial cap without touching proper
        nouns/acronyms). Length is unchanged, so caller char-bookkeeping is safe.
    """
    if not text:
        return "", text
    prev = (prev_display or "").rstrip()
    sentence_start = (not prev) or prev.endswith((".", "!", "?", ":", ";", "\n"))
    cased = text
    if cased[0].isalpha():
        first_word = cased.split(" ", 1)[0]
        if sentence_start:
            cased = cased[0].upper() + cased[1:]
        elif not first_word.isupper() and _norm_token(first_word) in _SEAM_LOWERCASE_WORDS:
            cased = cased[0].lower() + cased[1:]
    lead = " " if (prev_display and last_char and last_char not in _SEAM_NO_LEAD_AFTER) else ""
    return lead, cased


def _redecode_overlap(prev_text: str, new_text: str) -> bool:
    def _norm(tok: str) -> str:
        return tok.lower().strip(".,!?;:\"'")
    prev_toks = [t for t in (_norm(x) for x in prev_text.split()) if t]
    new_toks = [t for t in (_norm(x) for x in new_text.split()) if t]
    if len(prev_toks) < 2 or len(new_toks) < 2:
        return False

    # Pattern 1: tail-of-prev matches head-of-new (continuation re-decode).
    prev_tail = prev_toks[-6:]
    for k in (4, 3, 2):
        if len(new_toks) < k:
            continue
        new_head = new_toks[:k]
        for start in range(0, len(prev_tail) - k + 1):
            if prev_tail[start:start + k] == new_head:
                return True

    # Pattern 2: prefix re-decode â€” new and prev share a long common prefix
    # (whisper-stream re-emitting the same utterance with refined decoding,
    # e.g. "fox jumps over the laser" â†’ "fox jumps over the lazy dog").
    prefix = 0
    for a, b in zip(prev_toks, new_toks):
        if a == b:
            prefix += 1
        else:
            break
    if prefix >= 3 and prefix >= int(min(len(prev_toks), len(new_toks)) * 0.5):
        return True

    # Pattern 3: suffix re-decode â€” new is mostly contained at the end of prev
    # (e.g. prev "The quick brown fox jumps over the lazy dog",
    #       new  "The fox jumps over the lazy dog").
    # Require first-token match so we don't wipe out glued prefixes like
    # "Hello there How are you" when only "How are you" is re-emitted.
    if (
        len(new_toks) >= 3
        and len(new_toks) <= len(prev_toks)
        and prev_toks[0] == new_toks[0]
    ):
        tail_window = prev_toks[-(len(new_toks) + 2):]
        # count tokens of new_toks that appear in tail_window in order
        i = 0
        for tok in tail_window:
            if i < len(new_toks) and tok == new_toks[i]:
                i += 1
        if i >= max(3, int(len(new_toks) * 0.7)):
            return True

    return False


def _strip_whisper_hallucinations(text: str) -> str:
    stripped = text.strip()
    if not stripped:
        return ""

    # second-pass cleanup in case stream filter missed it
    stripped = re.sub(r"[\[\(]\s*(?:no\s*audio|blank[_\s]*audio|silence|music|applause|laughter|inaudible|background\s+noise)\s*[\]\)]", "", stripped, flags=re.IGNORECASE)
    stripped = re.sub(r"\.{2,}", ".", stripped)
    stripped = re.sub(r"\s+", " ", stripped).strip()
    if not stripped or re.fullmatch(r"[\s\.\,\-!?:;]+", stripped):
        return ""

    stripped = _collapse_repeated_sentences(stripped)

    stock_hit = False
    for pattern in _WHISPER_STOCK_PATTERNS:
        new_stripped, n = pattern.subn("", stripped)
        if n > 0:
            stock_hit = True
            stripped = new_stripped
    if stock_hit:
        stripped = re.sub(r"\s+", " ", stripped)
        stripped = re.sub(r"\s*[,.;:!?\-]+\s*$", "", stripped)
        stripped = re.sub(r"(?<=[.!?])\s*[,.;:!?\-]+", "", stripped)
        stripped = stripped.strip()

    if not stripped:
        return ""

    tokens = stripped.split()

    def _norm(tok: str) -> str:
        return tok.lower().strip(".,!?;:\"'")

    filtered = []
    for tok in tokens:
        rstripped = tok.rstrip(".,!?;:\"'")
        if len(rstripped) >= 2 and rstripped.endswith("-") and not rstripped.endswith("--"):
            continue
        filtered.append(tok)
    tokens = filtered

    cleaned = [t for t in (_norm(tok) for tok in tokens) if t]
    if cleaned and len(cleaned) <= 2 and all(tok in _DICTATION_HALLUCINATION_STOPWORDS for tok in cleaned):
        return ""

    deduped: list[str] = []
    for tok in tokens:
        key = _norm(tok)
        if deduped and key and key == _norm(deduped[-1]):
            continue
        deduped.append(tok)
    tokens = deduped

    while len(tokens) >= 3:
        tail = _norm(tokens[-1])
        if tail in _DICTATION_TRAILING_HALLUCINATIONS:
            tokens.pop()
        else:
            break
    return " ".join(tokens).strip()


@dataclass(frozen=True)
class ActionEvent:
    timestamp: float
    label: str
    display_text: str
    undoable: bool = False
    is_undo: bool = False


_UNDO_LABEL_PAIRS = {
    "spotify_next": "spotify_previous",
    "spotify_previous": "spotify_next",
    "spotify_toggle": "spotify_toggle",
    "spotify_shuffle": "spotify_shuffle",
    "spotify_repeat": "spotify_repeat",
    "youtube_next": "youtube_previous",
    "youtube_previous": "youtube_next",
    "youtube_toggle": "youtube_toggle",
    "chrome_back": "chrome_forward",
    "chrome_forward": "chrome_back",
}


# Two-hand pinch stretch sensitivity gain. The raw scale factor is
# (cur_palm_distance / anchor_palm_distance); we amplify the
# delta-from-1.0 by this multiplier before applying it to the
# overlay scale. With the velocity-adaptive palm smoother heavily
# damping slow motion to kill jitter (alpha=0.10 below 1.5% screen
# / frame), slow stretches felt sluggish without the gain. 1.6Ã—
# means a 10% palm-spread becomes ~16% canvas-stretch, which the
# user reported as the right responsiveness in testing.
_PINCH_SCALE_SENSITIVITY = 1.6


class _EngineRunner:
    # Runs engine.process_frame on a background Python thread so the
    # heaviest CPU step in the gesture loop (12-25 ms with MediaPipe,
    # 2-7 ms with ONNX/DirectML GPU) doesn't block the main-thread Qt
    # event loop. Without this decoupling, paint events from receivers
    # (mini viewer + live view, ~25-50 ms across both at 720p) push the
    # next QTimer fire out to 35-75 ms per cycle, capping `actual
    # self._fps` at 16-26 even when per-tick wall time is healthy.
    #
    # MediaPipe and ONNX both release the GIL inside their C++/native
    # inference paths, so we get true parallelism across the engine
    # call and the main thread's post-processing.
    #
    # Coordination: caller checks `busy` before submitting; caller
    # holds an external "in-flight" view by reading `busy` and
    # treating True as back-pressure (skip this tick). When inference
    # finishes, the runner invokes `result_callback(frame, result)` on
    # the runner thread â€” the callback is responsible for getting back
    # to the GUI thread (typically by emitting a Qt signal connected
    # via auto/queued connection).
    #
    # `set_engine` is guarded by a lock so mid-session engine swaps
    # (Lite Mode toggle, GPU Mode toggle) don't race with an in-flight
    # process_frame call.
    def __init__(self, name: str = "GestureEngineRunner") -> None:
        self._name = name
        self._request_queue: "queue.Queue[Optional[np.ndarray]]" = queue.Queue(maxsize=1)
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._engine_lock = threading.Lock()
        self._current_engine: GestureRecognitionEngine | None = None
        self._busy = False
        self._result_callback = None
        self._post_infer = None

    @property
    def busy(self) -> bool:
        return self._busy

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self, result_callback) -> None:
        if self.is_running:
            return
        self._result_callback = result_callback
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run, daemon=True, name=self._name
        )
        self._thread.start()

    def stop(self, timeout: float = 2.0) -> None:
        self._stop_event.set()
        try:
            self._request_queue.put_nowait(None)
        except queue.Full:
            pass
        thread = self._thread
        self._thread = None
        if thread is not None:
            # Pump the Qt event loop in 50 ms slices while waiting
            # for the engine thread to finish so the UI thread (which
            # calls this from stop_engine / start_engine on a hot
            # camera swap) keeps painting the camera viewers and
            # processing input. Without the pumps the viewer freezes
            # for up to `timeout` seconds on every swap.
            try:
                from PySide6.QtWidgets import QApplication as _QApp
                _qapp = _QApp.instance()
            except Exception:
                _qapp = None
            if _qapp is not None:
                elapsed = 0.0
                step = 0.05
                while elapsed < timeout and thread.is_alive():
                    thread.join(timeout=step)
                    elapsed += step
                    try:
                        _qapp.processEvents()
                    except Exception:
                        pass
            else:
                thread.join(timeout=timeout)
        self._busy = False
        self._result_callback = None

    def set_engine(self, engine: "GestureRecognitionEngine | None") -> None:
        with self._engine_lock:
            self._current_engine = engine

    def submit(self, frame) -> bool:
        if self._busy or not self.is_running:
            return False
        try:
            self._request_queue.put_nowait(frame)
        except queue.Full:
            return False
        self._busy = True
        return True

    def _run(self) -> None:
        # Rolling samples of (real_inference_seconds, found_hand_bool)
        # so we can attribute the "engine" timing in the main-thread
        # diagnostic to true inference vs main-thread queue-wait.
        # Logged every ~2 s alongside hand-in-frame ratio.
        log_samples: list[tuple[float, bool]] = []
        last_log_at = time.monotonic()
        while not self._stop_event.is_set():
            try:
                frame = self._request_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            if frame is None:
                break
            result = None
            inference_start = time.perf_counter()
            try:
                # Hold the engine lock for the entire process_frame
                # call â€” main-thread mode-swap callers acquire the
                # same lock via set_engine(), so they will block until
                # this in-flight inference returns. That guarantees
                # the engine the swap is closing isn't being touched
                # at the moment of close().
                with self._engine_lock:
                    engine = self._current_engine
                    if engine is not None:
                        result = engine.process_frame(frame)
            except Exception:
                traceback.print_exc()
                result = None
            finally:
                self._busy = False
            # GPU/Lite custom-gesture MediaPipe must not run on the
            # GUI thread — it was pulling camera display from ~50 fps
            # toward ~30. Attach landmarks here so _on_engine_result
            # only classifies.
            post = self._post_infer
            if post is not None and result is not None:
                try:
                    post(frame, result)
                except Exception:
                    traceback.print_exc()
            inference_seconds = time.perf_counter() - inference_start
            found = bool(getattr(result, "found", False)) if result is not None else False
            log_samples.append((inference_seconds, found))
            now = time.monotonic()
            if now - last_log_at >= 2.0 and len(log_samples) >= 8:
                last_log_at = now
                with_hand = [s[0] for s in log_samples if s[1]]
                no_hand = [s[0] for s in log_samples if not s[1]]
                try:
                    msg = (
                        f"[engine_runner] real-inference avg "
                        f"hand={(sum(with_hand) / len(with_hand) * 1000.0) if with_hand else 0.0:.1f}ms "
                        f"({len(with_hand)} samples) "
                        f"no_hand={(sum(no_hand) / len(no_hand) * 1000.0) if no_hand else 0.0:.1f}ms "
                        f"({len(no_hand)} samples)\n"
                    )
                    sys.stderr.write(msg)
                    sys.stderr.flush()
                except Exception:
                    pass
                log_samples.clear()
            cb = self._result_callback
            if cb is not None:
                try:
                    cb(frame, result)
                except Exception:
                    traceback.print_exc()


# Fire-once static poses (open Chrome / Spotify / Touchless, mode
# toggles, voice, clip) must be held this long before the action
# runs. Continuous poses (volume, wheels, mouse-once-on) keep their
# own timings.
_STATIC_GESTURE_HOLD_SECONDS = 1.0


class GestureWorker(QObject):
    status_changed = Signal(str)
    command_detected = Signal(str)
    # Granular debug-log events that don't map cleanly to a finished
    # action. Examples emitted today: "Voice heard: '<text>'",
    # "Voice command not recognized", "Spotify not running — launching".
    # The home page's debug log subscribes via _on_engine_log_message
    # and pipes these into the same widget command_detected feeds.
    engine_log = Signal(str)
    camera_selected = Signal(str)
    error_occurred = Signal(str)
    running_state_changed = Signal(bool)
    debug_frame_ready = Signal(object, object)
    save_prompt_completed = Signal(object)
    action_history_changed = Signal(object)
    dictation_stream_ready = Signal(str)
    # Fired on every off->on transition of mouse mode so the main
    # window can pop a "choose which monitor to control" picker
    # before the user starts moving the cursor. Receivers can skip
    # the popup based on config.mouse_active_monitor_index already
    # being set — engine doesn't make that decision; UI does.
    mouse_mode_activated = Signal()
    # Fired when the open_touchless gesture binding triggers. Main
    # window connects a slot that does show + raise_ + activateWindow
    # so the user can hand-gesture-summon Touchless back to focus
    # without alt-tabbing.
    open_touchless_requested = Signal()
    # Fired when the instant_clip gesture binding triggers. Main
    # window connects a slot that immediately exports the configured
    # clip duration ending at the current moment, saved to
    # clips_save_dir with a timestamped filename — NO prompt, NO
    # confirmation dialog. Same as the voice "clip that" command,
    # but bound to a single gesture for hands-only operation.
    instant_clip_requested = Signal()
    # Decoupled display path. Emitted from `_tick` immediately after
    # the camera read + flip + prepare step, BEFORE the engine
    # dispatch. Receivers connect to this for the live-view paint
    # so display latency is "camera frame interval + paint" rather
    # than "camera + engine + post-engine + paint". `debug_frame_ready`
    # still fires later with the engine-annotated frame for callers
    # that want it (legacy compatibility), but the live-view
    # widgets ignore the frame in `debug_frame_ready` and only use
    # its payload for text widgets / state.
    # Carries (frame_bgr, capture_monotonic_ts). The timestamp is the
    # `time.monotonic()` value at the moment the reader thread
    # finished decoding this frame â€” receivers subtract it from
    # `time.monotonic()` at paint time to get end-to-end pipeline
    # latency (camera â†’ display).
    raw_frame_ready = Signal(object, float)
    # Engine-extracted landmark coordinates per hand. Each hand is a
    # list of (x, y) tuples in [0, 1] normalized image space â€” same
    # space the cv2 overlay code used. Receivers feed these into
    # GpuVideoWidget.update_landmarks for GPU-side overlay rendering;
    # we no longer draw landmarks on the BGR frame on the CPU.
    engine_landmarks_ready = Signal(object)
    # Fired when the user opens/focuses Spotify by gesture or voice while NOT
    # connected to the Spotify Web API — the main window shows a "connect
    # Spotify" pill in response. Throttled by the worker.
    spotify_connect_prompt_requested = Signal()
    # Internal: runner-thread â†’ main-thread bridge for engine results.
    # Emitted from the _EngineRunner thread when inference completes;
    # the Auto/Queued connection delivers the slot call on the
    # GestureWorker's owning (main) thread, so all post-processing â€”
    # including overlay widget mutations â€” stays on the GUI thread.
    _engine_result_ready = Signal(object, object)
    # Pipeline freeze state. Receivers (live-view widgets) use this
    # to render a "Paused â€” recording custom gesture" overlay with
    # a light blur so the user knows the main app intentionally
    # stopped reacting while a recording window is open.
    frozen_state_changed = Signal(bool)
    # Custom-gesture "show_overlay_drawing" action: emitted when a
    # bound gesture fires, carrying the user-typed filename. The
    # main window resolves it against the configured drawings dir
    # and toggles a transparent always-on-top overlay window.
    drawing_overlay_toggle_requested = Signal(str)
    # Pinch-grab transform updates for the visible drawing overlay.
    # Carries absolute (cumulative-since-overlay-shown) values
    # â€” dx_norm, dy_norm in normalised screen units, and scale as a
    # multiplier on the auto-fit base â€” so the main window forwards
    # them straight to DrawingOverlayWindow.set_grab_transform with
    # no further state.
    drawing_overlay_grab_transform = Signal(float, float, float)
    # True while at least one hand is actively pinching, False when
    # the gesture releases. Lets the main window show / hide a
    # subtle "grabbing" affordance without polling.
    drawing_overlay_grab_active = Signal(bool)

    _LOW_FPS_AUTO_THRESHOLD = 18.0
    _LOW_FPS_AUTO_ENTER_SECONDS = 4.0
    _LOW_FPS_AUTO_EXIT_SECONDS = 6.0
    # CRITICAL-FPS auto-engage: on slow desktops (no fullscreen game),
    # engage low-fps mode anyway when FPS stays below 12 for 6 s. This
    # rescues users on older hardware (GTX 960 / similar) who get ~7 fps
    # by default and would otherwise need to manually find + toggle
    # Lite Mode. Exit threshold uses ASYMMETRIC hysteresis: must stay
    # above 28 fps for 8 s before disengaging, otherwise the bump from
    # lite_mode (which lifts 7 → ~25 fps on such hardware) would oscillate
    # right at the normal-mode threshold of 18 fps.
    _CRITICAL_FPS_THRESHOLD = 12.0
    _CRITICAL_FPS_ENTER_SECONDS = 6.0
    _CRITICAL_FPS_EXIT_THRESHOLD = 28.0
    _CRITICAL_FPS_EXIT_SECONDS = 8.0
    _FORCED_TEST_FPS_TARGET = 10.0
    _NORMAL_PROCESS_WIDTH = 960
    _LOW_FPS_PROCESS_WIDTH = 384
    # Lite Mode runs the lite landmark model on a downsampled
    # inference frame. MediaPipe's palm detector internally rescales
    # to a 192-px square ROI for landmark inference, so dropping the
    # input from 960 â†’ 384 px gives the palm detector ~6x less
    # pixels to scan without harming landmark accuracy. Same width
    # Low-FPS mode uses, with the difference that Lite keeps
    # Normal-mode confidence thresholds + stable-frame requirement
    # so the gesture decisions still feel as solid as before.
    _LITE_MODE_PROCESS_WIDTH = 384
    # Darwin GPU: ONNX/CoreML plus a 480-px frame. Default GPU used
    # 960 px on 720p — same work as Default, so fps stayed ~13-14.
    _MAC_GPU_PROCESS_WIDTH = 480
    _FULLSCREEN_POLL_INTERVAL = 1.0
    # Drawing pen-lift hold duration. When the user opens their
    # thumb, the pen lifts after this many seconds of continuous
    # open-thumb detection. 0.10 s (~3 frames at 30 fps) feels
    # near-instant; the strict 4-condition open-thumb gate below
    # plus the freeze-pen (which locks the stroke endpoint during
    # the hold) keep brief rotation wobble from false-firing even
    # at this shorter hold.
    _DRAWING_THUMB_OPEN_HOLD_SECONDS = 0.10
    # One Euro filter params for the drawing fingertip cursor. Applied
    # to the RAW normalized fingertip (before the control-box remap) so
    # tuning is independent of box gain. Lower min_cutoff => steadier
    # when holding still; higher beta => less lag on fast strokes.
    # min_cutoff lowered 1.2 -> 0.6: at rest the cursor was still drifting
    # / shaking (hand tremor leaking through); 0.6 Hz holds it much steadier
    # when the hand is held still. beta keeps fast strokes responsive.
    # r53 v2: min_cutoff 0.6 -> 0.35. After the beta drop, residual
    # drift after motion was still visible because at-rest cutoff was
    # still 0.6 Hz — hand tremor and MediaPipe jitter leaked through
    # for a beat after the user stopped. 0.35 doubles at-rest
    # smoothing without pushing lag noticeably higher.
    _DRAWING_OEF_MIN_CUTOFF = 0.35
    # r53 v6: beta 0.15 -> 0.35 to unstick the cursor on fast
    # strokes. 0.15 was too fixed-low-pass — every quick swipe or
    # letter loop lagged behind the finger by 60-100 ms and the
    # cursor didn't fully catch up. 0.35 is between yesterday's
    # 0.6 (jaggy curves) and my r53 0.15 (laggy fast): OneEuro's
    # cutoff climbs enough with velocity that fast strokes track
    # in real time, but the base 0.35 Hz cutoff still gives clean
    # steady-state circles at slow speeds.
    _DRAWING_OEF_BETA = 0.35
    # r53 v2: minimum pixel movement before a new stroke sample is
    # committed. Kills residual drift when the hand is held still —
    # any point closer than this to the previously committed point
    # is dropped, so hand tremor at rest doesn't turn into ink.
    _DRAWING_MIN_MOVE_PX = 3.0
    # Suggestion overlay: triggered when measured FPS stays below 15 for
    # longer than 10 seconds. After the user dismisses (X, left-fist, or
    # auto-dismiss), we wait this many seconds before re-offering.
    _LOW_FPS_SUGGEST_THRESHOLD = 15.0
    _LOW_FPS_SUGGEST_ENTER_SECONDS = 10.0
    _LOW_FPS_SUGGEST_COOLDOWN_SECONDS = 300.0

    def __init__(self, config, camera_index_override: Optional[int] = None, parent=None, progress_callback=None):
        super().__init__(parent)
        # Helper: pumps the Qt event loop briefly so the
        # starting-pill's repaint timer can fire between heavy
        # controller / overlay constructions below. Without these
        # pumps the entire __init__ is one ~2 s blocking Python
        # call and the starting pill's progress bar can't be
        # repainted until __init__ returns. Each pump_events() call
        # also advances the optional progress_callback so the bar
        # steps forward by one checkpoint per pump.
        self._init_progress_step = 0
        self._init_progress_total = 10  # ~9 pumps in __init__ + headroom
        self._init_progress_callback = progress_callback

        def _pump_events() -> None:
            self._init_progress_step += 1
            cb = self._init_progress_callback
            if cb is not None:
                try:
                    fraction = min(0.85, self._init_progress_step / float(self._init_progress_total))
                    cb(fraction)
                except Exception:
                    pass
            try:
                from PySide6.QtWidgets import QApplication as _QApp
                app = _QApp.instance()
                if app is not None:
                    app.processEvents()
            except Exception:
                pass

        self._pump_events = _pump_events
        self.config = config
        self.camera_index_override = camera_index_override
        self._running = False
        # Pipeline freeze: when True, _tick still emits raw_frame_ready
        # so subscribers (custom-gesture recorder) keep receiving
        # camera frames, but skips the entire engine pipeline â€”
        # MediaPipe inference, gesture-action dispatch, all of it.
        # Set via set_pipeline_frozen() while a modal recording window
        # is open so we don't double-run MediaPipe (the recorder runs
        # its own pass) and so gesture actions don't fire while the
        # user is intentionally posing for sample capture.
        self._frozen = False
        # Pinch-grab state for moving / scaling a visible drawing
        # overlay. Modes: "none" (no pinch), "one" (single-hand
        # translate), "two" (bimanual translate + scale). Mode
        # transitions lock in the previous mode's contribution to
        # _pinch_accum_* before starting a fresh anchor.
        self._pinch_mode: str = "none"
        # Which palm slot is driving the current single-hand grab.
        # "primary" or "secondary" depending on which hand the user
        # is pinching with. Lets the user grab with EITHER hand
        # (left alone, right alone, or both for stretch). Switching
        # hands mid-grab is treated as a mode transition so the
        # accumulated offset doesn't teleport.
        self._pinch_one_slot: Optional[str] = None
        self._pinch_anchor_palm: Optional[tuple[float, float]] = None  # one-hand
        self._pinch_two_anchor_dist: float = 0.0
        self._pinch_two_anchor_mid: tuple[float, float] = (0.0, 0.0)
        # Cumulative transform applied to the overlay since it was
        # shown. Live mode transforms add on top; mode transitions
        # bake the live delta into these.
        self._pinch_accum_dx: float = 0.0
        self._pinch_accum_dy: float = 0.0
        self._pinch_accum_scale: float = 1.0
        # Sticky-active grace. MediaPipe landmark output for a
        # foreshortened pinch can flicker stable_label between
        # 'pinch' and a competing label for one or two frames at
        # a time, which dropped the grab mode to 'none' and forced
        # an anchor reset every flicker â€” the user-visible result
        # was the drawing snapping back to its baked offset on
        # every dropped frame. We now record the last time we saw
        # the pinch label and treat the user as still pinching for
        # `_pinch_grace_seconds` of label silence so brief
        # recogniser misses don't cancel the grab.
        self._pinch_grace_seconds: float = 0.4
        self._pinch_last_seen_primary: float = 0.0
        self._pinch_last_seen_secondary: float = 0.0
        # Pre-activation hold: a slot has to be in pinch pose for
        # this many seconds CONTINUOUSLY (within grace tolerance)
        # before grab mode actually engages. Stops the drawing
        # from jumping the moment a transient frame stabilises to
        # 'pinch' â€” the user reported feeling like the grab kicked
        # in 'before I'm actually in pinch'. _pinch_streak_start_*
        # records when the current pinch streak began (0.0 when no
        # streak is in progress).
        self._pinch_activation_delay: float = 0.7
        self._pinch_streak_start_primary: float = 0.0
        self._pinch_streak_start_secondary: float = 0.0
        # EMA-smoothed palm positions used for grab math. Raw
        # palm.center coords jitter several pixels per frame on
        # foreshortened poses; passing the raw values straight to
        # the transform makes the displayed drawing visibly
        # shake. The smoother only runs while a pinch is active
        # so neutral motion isn't artificially laggy.
        self._pinch_smoothed_primary: Optional[tuple[float, float]] = None
        self._pinch_smoothed_secondary: Optional[tuple[float, float]] = None
        # 0.0 â†’ output never changes (full damping); 1.0 â†’ no
        # damping. 0.35 keeps the response feel-snappy while
        # killing single-frame jitter.
        self._pinch_smooth_alpha: float = 0.35
        self._cap = None
        self._camera_info = None
        self.engine: GestureRecognitionEngine | None = None
        self._last_status_text = ""
        self._last_spotify_action = "-"
        self._last_chrome_action = "-"
        self._last_time = time.time()
        self._fps = 0.0
        # macOS perf diagnostic: when HGR_PERF_LOG=1 the per-frame timing
        # attribution ([lite_mode/timing] read/prep/engine/vol/app/wheel/
        # overlay/emit ms) is emitted in ALL modes, not just Lite/GPU, so a
        # 3-fps report on Mac can be attributed to capture vs paint vs
        # inference without toggling a perf mode. Off by default → zero cost.
        self._perf_log_enabled = os.getenv("HGR_PERF_LOG", "").strip() == "1"
        self._dynamic_hold_label = "neutral"
        self._dynamic_hold_until = 0.0
        # Last static / dynamic labels we emitted a `gesture_detected`
        # telemetry event for, per hand. Fires on transitions so a
        # held pose doesn't spam one event per frame.
        self._telemetry_last_static_label = "neutral"
        self._telemetry_last_dynamic_label = "neutral"
        self._telemetry_last_static_label_secondary = "neutral"
        self._telemetry_last_dynamic_label_secondary = "neutral"
        # Debounce: timestamp of the last `gesture_detected` emit per
        # (handedness, gesture) tuple. Suppresses recognizer flicker
        # (rapid pinch→neutral→pinch cycles) that would otherwise
        # over-count transitions on the secondary hand.
        self._telemetry_gesture_last_emit: dict[tuple[str, str], float] = {}
        self._low_fps_active = bool(getattr(config, "low_fps_mode", False))
        self._low_fps_below_since: float | None = None
        self._low_fps_above_since: float | None = None
        self._low_fps_auto_engaged = False
        self._low_fps_last_process = 0.0
        # Track LAST APPLIED mode state separately from config. The UI
        # updates config.lite_mode (etc.) BEFORE calling set_*_mode on
        # the worker, so comparing was_enabled = bool(config.lite_mode)
        # against the new value ALWAYS shows no-op — the no-op gate
        # silently swallowed every mode toggle. Initialize to None so
        # the first toggle always applies (even when matching the
        # persisted config value at startup).
        self._applied_lite_mode: bool | None = None
        self._applied_gpu_mode: bool | None = None
        self._applied_low_fps_mode: bool | None = None
        self._applied_mac_performance_boost: bool | None = None
        # r54 v9: pre-flight short-shutter idempotence latch. Tracks
        # which device_name the pre-flight fired for so we don't
        # re-arm on every camera-path re-check. Cleared on ffmpeg
        # teardown (want_ffmpeg=False branch).
        self._ffmpeg_preflight_device: str | None = None
        # Suggestion overlay bookkeeping (separate from auto-engage timing).
        self._low_fps_suggest_below_since: float | None = None
        self._low_fps_suggest_cooldown_until = 0.0
        self._low_fps_suggest_visible = False
        self._fullscreen_foreground_active = False
        self._fullscreen_foreground_process = ""
        self._fullscreen_check_last = 0.0
        # v1.1.7 event-loop optimization (Step 2): when a fullscreen
        # app (typically a game) has foreground and the user has GPU
        # inference enabled (config.gpu_mode OR the implicit
        # prefer_gpu=True baked into Lite Mode), auto-suppress the
        # GPU path and fall back to CPU MediaPipe for the duration of
        # the game session. DirectML inference and the game's GPU
        # work were fighting for the same GPU, dragging both. When
        # the game closes / minimizes, the transition-detection in
        # _refresh_fullscreen_foreground clears this flag and swaps
        # the engine back to the GPU path automatically. Transparent
        # to the user — no config toggle, no dialog.
        self._gpu_suppressed_for_fullscreen: bool = False
        # C17 (v1.1.7): engine cache keyed by config signature. Mode
        # switches (Lite ↔ GPU, GPU ↔ default, etc.) used to rebuild
        # the HandDetector every time, which re-loaded the ONNX +
        # DirectML session and blocked the main thread for ~2 seconds.
        # On a GPU-capable box that runs both modes on the same ONNX
        # model, the second build is pure re-init cost. Caching by
        # signature makes repeat toggles instant (cache HIT) and keeps
        # first-time builds at their existing cost. Bounded at ~4-6
        # unique configs in practice; cleared in _shutdown_runtime.
        self._engine_cache: "dict[tuple, GestureRecognitionEngine] | None" = None
        # Phone-camera-via-QR capture (owned by MainWindow / PhoneCameraServer).
        # None when no phone is connected; set by set_phone_camera_capture().
        self._phone_camera_capture = None

        self.low_fps_suggestion_overlay = LowFpsSuggestionOverlay()
        self.low_fps_suggestion_overlay.activateRequested.connect(self._handle_low_fps_suggestion_activate)
        self.low_fps_suggestion_overlay.dismissed.connect(self._handle_low_fps_suggestion_dismissed)

        _pump_events()
        self.volume_controller = VolumeController()
        self.volume_overlay = ScreenVolumeOverlay(config)
        self.volume_overlay.attach_controller(self.volume_controller)
        _pump_events()
        self.voice_status_overlay = VoiceStatusOverlay(config)
        try:
            self.voice_status_overlay.selectionChosen.connect(self._handle_voice_overlay_selection)
        except Exception:
            pass
        self.dictation_stream_ready.connect(self._on_dictation_stream_ready)
        self.volume_tracker = VolumeGestureTracker()
        self._volume_message = self.volume_controller.message
        self._volume_mode_active = False
        self._volume_level: float | None = self.volume_controller.get_level()
        self._volume_status_text = "idle"
        # Mute cache must be initialised before the first call to
        # _read_system_mute() â€” that helper reads these fields and
        # sets the cache, and gets called as part of constructing
        # _volume_muted on the very next line.
        self._mute_cache_value: bool = False
        self._mute_cache_until: float = 0.0
        self._volume_muted = self._read_system_mute()
        self._volume_overlay_visible = False
        self._mute_block_until = 0.0
        self._volume_dual_active = False
        self._volume_app_level: float | None = None
        self._volume_app_label = ""
        self._volume_app_process = ""
        self._volume_bar_selected = "sys"
        self._volume_init_palm_x: float | None = None
        self._volume_app_check_until = 0.0
        self._volume_nudge_next_at = 0.0
        self._volume_nudge_last_dir = 0
        self._youtube_volume_step_next_at = 0.0
        # Hold-and-fire state for the right-hand plain-three / plain-four
        # actions (open_chrome / open_touchless). These bindings have
        # no router consuming them — fired here directly from
        # _maybe_fire_open_chrome_touchless on a 1.0 s hold.
        self._open_action_candidate: Optional[str] = None
        self._open_action_candidate_since: float = 0.0
        self._open_action_cooldown_until: float = 0.0
        # Rate-limited diagnostic: emit the right-hand stable_label
        # once every ~1 s so the user can read the debug log while
        # testing without flooding it at frame rate.
        self._open_action_last_diag_label: str = ""
        self._open_action_last_diag_at: float = 0.0
        # YouTube auto-pause-on-absence state. Tracks the wall-clock
        # time we first noticed no hand in frame (0.0 = currently
        # visible). When this exceeds the configured threshold, we
        # fire toggle_playback() once and latch _youtube_paused_for_absence
        # so we don't keep re-pausing. On hand return, if we latched
        # earlier, we fire toggle_playback() once to resume.
        self._youtube_absence_started_at: float = 0.0
        self._youtube_paused_for_absence: bool = False
        # YouTube auto-skip-ads background tick. The per-frame engine
        # callback would poll at frame rate (~30 Hz) which wastes CPU
        # on template matching; throttle to once per 1.0 s.
        self._youtube_auto_skip_last_tick: float = 0.0
        self._chrome_active_cache = False
        self._chrome_active_cache_until = 0.0
        self._spotify_active_cache = False
        self._spotify_active_cache_until = 0.0
        # Stale-while-revalidate flag for the wheel-active probe.
        # is_active_for_wheel() can fall through to a Spotify Web API
        # call when the desktop window isn't visible yet (Spotify is
        # still launching), and that HTTP call can block up to 5 s.
        # Running it on the worker thread froze the camera display
        # for the duration of the launch. Now we kick the refresh on
        # a background thread and return the previous cached value.
        self._spotify_active_refresh_in_flight = False
        self._chrome_wheel_candidate = "neutral"
        self._chrome_wheel_candidate_since = 0.0
        self._chrome_wheel_visible = False
        self._chrome_wheel_anchor = None
        self._chrome_wheel_selected_key: str | None = None
        self._chrome_wheel_selected_since = 0.0
        self._chrome_wheel_pose_grace_until = 0.0
        self._chrome_wheel_cooldown_until = 0.0
        self._chrome_wheel_cursor_offset: tuple[float, float] | None = None
        self._spotify_wheel_candidate = "neutral"
        self._spotify_wheel_candidate_since = 0.0
        self._spotify_wheel_visible = False
        self._spotify_wheel_anchor = None
        self._spotify_wheel_selected_key: str | None = None
        self._spotify_wheel_selected_since = 0.0
        self._spotify_wheel_pose_grace_until = 0.0
        self._spotify_wheel_cooldown_until = 0.0
        self._spotify_wheel_cursor_offset: tuple[float, float] | None = None
        self._youtube_wheel_candidate = "neutral"
        self._youtube_wheel_candidate_since = 0.0
        self._youtube_wheel_visible = False
        self._youtube_wheel_anchor = None
        self._youtube_wheel_selected_key: str | None = None
        self._youtube_wheel_selected_since = 0.0
        self._youtube_wheel_pose_grace_until = 0.0
        self._youtube_wheel_cooldown_until = 0.0
        self._youtube_wheel_cursor_offset: tuple[float, float] | None = None
        _pump_events()
        self.spotify_wheel_overlay = SpotifyWheelOverlay(config)
        self.chrome_wheel_overlay = SpotifyWheelOverlay(config)
        self.youtube_wheel_overlay = SpotifyWheelOverlay(config)
        _pump_events()
        self.mouse_controller = MouseController()
        self.mouse_tracker = self._build_mouse_tracker()
        self._last_mouse_update = SimpleNamespace(
            mode_enabled=False,
            cursor_position=None,
            left_click=False,
            left_press=False,
            left_release=False,
            right_click=False,
            scroll_steps=0,
        )
        self._mouse_mode_enabled = False
        self._mouse_control_text = "Mouse mode off" if self.mouse_controller.available else self.mouse_controller.message
        self._mouse_status_text = "off" if self.mouse_controller.available else "unavailable"

        _pump_events()
        self.chrome_controller = ChromeController()
        self.chrome_router = ChromeGestureRouter(static_hold_seconds=_STATIC_GESTURE_HOLD_SECONDS, static_cooldown_seconds=1.5, dynamic_cooldown_seconds=1.5)
        self._chrome_mode_enabled = False
        self._chrome_control_text = self.chrome_controller.message

        _pump_events()
        self.spotify_controller = SpotifyController()
        # Surface the "connect Spotify" pill when the user opens Spotify by
        # gesture/voice without having connected the Web API (throttled).
        self._spotify_connect_prompt_cooldown_until = 0.0
        self.spotify_controller.set_opened_listener(self._request_spotify_connect_prompt)
        self.spotify_router = SpotifyGestureRouter(static_hold_seconds=_STATIC_GESTURE_HOLD_SECONDS, static_cooldown_seconds=1.5, dynamic_cooldown_seconds=1.5)
        self._spotify_control_text = self.spotify_controller.message

        _pump_events()
        # Discord controller — owns the local RPC pipe to the Discord
        # desktop client (mute / deafen / read voice state). Used by
        # both the Settings → Connect button auth flow and the
        # discord_router below for gesture-driven control.
        self.discord_controller = DiscordController()
        self.discord_router = DiscordGestureRouter(
            static_hold_seconds=_STATIC_GESTURE_HOLD_SECONDS,
            static_cooldown_seconds=1.5,
            toggle_hold_seconds=_STATIC_GESTURE_HOLD_SECONDS,
            toggle_cooldown_seconds=1.5,
        )
        self._discord_control_text = self.discord_controller.message
        self._discord_mode_prev_active = False
        self._last_discord_action_counter = 0

        _pump_events()
        self.youtube_controller = YouTubeController(volume_controller=self.volume_controller)
        self.youtube_router = YouTubeGestureRouter(
            static_hold_seconds=_STATIC_GESTURE_HOLD_SECONDS,
            static_cooldown_seconds=1.5,
            dynamic_cooldown_seconds=1.0,
            toggle_hold_seconds=_STATIC_GESTURE_HOLD_SECONDS,
        )
        self._youtube_control_text = "YouTube idle"
        self._youtube_mode_info = "off"
        self._youtube_mode_prev_active = False
        self._chrome_mode_prev_active = False
        self._last_chrome_action_counter = 0
        self._last_spotify_action_counter = 0
        self._last_youtube_action_counter = 0

        self._action_history: deque[ActionEvent] = deque(maxlen=20)
        self._action_history_lock = threading.Lock()
        self._volume_session_prev_level: float | None = None
        self._volume_session_prev_app_level: float | None = None
        self._spotify_info_text = "-"
        self._spotify_vol_lock = threading.Lock()
        self._spotify_vol_target: int | None = None
        self._spotify_vol_last_sent: int | None = None
        self._spotify_vol_worker: threading.Thread | None = None

        _pump_events()
        self.voice_listener = VoiceCommandListener(
            preferred_input_device=getattr(config, "preferred_microphone_name", None),
            input_gain=getattr(config, "mic_input_gain", 1.0),
            input_gain_auto=bool(getattr(config, "mic_input_gain_auto", True)),
        )
        self.voice_processor = VoiceCommandProcessor(
            chrome_controller=self.chrome_controller,
            spotify_controller=self.spotify_controller,
            discord_controller=self.discord_controller,
            # Step-emit callback. Every meaningful step inside the
            # processor's multi-step commands (find channel → focus
            # window → paste → press enter) gets forwarded to the
            # engine_log signal so the user sees granular progress
            # in the detailed log surface. Wrapped in try/except via
            # _emit_voice_log_step to keep logging failures from
            # breaking the underlying command.
            log_step=self._emit_voice_log_step,
        )
        _pump_events()
        self.live_dictation_streamer = LiveDictationStreamer(
            preferred_microphone_name=getattr(config, "preferred_microphone_name", None),
        )
        self.text_input_controller = TextInputController()
        self.dictation_processor = DictationProcessor()
        _pump_events()
        self.llama_server = LlamaServer()
        try:
            print(f"[hgr] llama_server: available={self.llama_server.available} backend={self.llama_server.backend} message={self.llama_server.message}")
        except Exception:
            pass
        self.grammar_corrector = GrammarCorrector(server=self.llama_server)
        import os as _os_refiner
        # Refiner is opt-IN: it duplicates text when streamer emits multiple
        # finals mid-utterance. Set HGR_ENABLE_WHISPER_REFINER=1 to test.
        self._refiner_enabled = _os_refiner.getenv("HGR_ENABLE_WHISPER_REFINER", "").strip() in {"1", "true", "yes"}
        if self._refiner_enabled:
            self.whisper_refiner = WhisperRefiner(on_refinement=self._apply_refinement)
            try:
                print(f"[hgr] whisper_refiner: available={self.whisper_refiner.available} backend={self.whisper_refiner.backend} message={self.whisper_refiner.message}")
            except Exception:
                pass
        else:
            self.whisper_refiner = None  # type: ignore[assignment]
            print("[hgr] whisper_refiner: disabled via HGR_DISABLE_WHISPER_REFINER")
        self._corrector_pending_text = ""
        self._corrector_lock = threading.Lock()
        self._corrector_applied_message = ""
        self._dictation_state: dict | None = None
        self._app_hint_thread: threading.Thread | None = None
        self._prime_voice_app_hints_async()
        self._prime_voice_runtime_async()
        self._voice_control_text = self.voice_listener.message
        self._voice_heard_text = "-"
        self._voice_candidate = "neutral"
        self._voice_candidate_since = 0.0
        self._voice_cooldown_until = 0.0
        self._voice_latched_label: str | None = None
        self._voice_one_two_triggered_at: float = 0.0
        # Palm anchor for the left-hand "one" hold. Used to reject
        # swipe-scale translation (custom 1-swipe-left must not start
        # voice) while still allowing jitter / a little drift.
        self._voice_hold_origin_xy: Optional[tuple[float, float]] = None
        self._voice_hold_origin_scale: float = 1.0
        # Left-hand thumbs-up CLIP GESTURE state. Hold thumbs-up on
        # the left hand for 0.5 s to trigger a clip with the user's
        # configured default duration (Settings → Clip Presets).
        # Mirrors the voice "clip that" pipeline; uses the same
        # _pending_clip_voice_end_ts anchor so the saved clip ends
        # at the moment of gesture detection. Cooldown prevents
        # accidental double-trigger from a held pose flicker.
        self._clip_gesture_candidate_since: float = 0.0
        self._clip_gesture_cooldown_until: float = 0.0
        self._left_hand_prediction = None
        # Cached LEFT-hand reading (landmarks + finger states). Set
        # alongside _left_hand_prediction each frame so other systems
        # can inspect specific finger geometry without re-running
        # MediaPipe. Used by Discord's MRP activator pose detector —
        # the recogniser labels any three-finger pose as "three", so
        # to distinguish "index+middle+ring" (mouse-mode toggle) from
        # "middle+ring+pinky" (Discord-mode toggle) we need the
        # actual per-finger states.
        self._left_hand_reading = None
        self._left_hand_streak_since = 0.0
        self._voice_queue: queue.Queue[tuple[int, object]] = queue.Queue()
        self._voice_thread: threading.Thread | None = None
        self._voice_request_id = 0
        self._voice_stop_event: threading.Event | None = None
        self._voice_listening = False
        self._dictation_active = False
        self._dictation_backend = "idle"
        self._voice_mode = "ready"
        self._voice_display_text = "-"
        self._dictation_toggle_release_required = False
        self._dictation_stop_rearm_at = 0.0
        self._dictation_release_candidate_since = 0.0
        self._tutorial_mode_enabled = False
        self._tutorial_step_key: str | None = None
        # Tutorial-volume-step gate: when True, drop the shaka-mute
        # trigger so the OS audio doesn't actually mute until the
        # tutorial advances past its 'practise up/down first' phase.
        # Tutorial flips this on entering the volume step and off
        # once swing-up + swing-down are both complete.
        self._tutorial_block_mute: bool = False
        self._drawing_mode_enabled = False
        self._drawing_toggle_candidate_since = 0.0
        self._drawing_toggle_cooldown_until = 0.0
        self._drawing_cursor_norm: tuple[float, float] | None = None
        self._drawing_tool = "hidden"
        self._drawing_control_text = "Drawing mode off"
        self._drawing_render_target = "screen"
        self._drawing_brush_hex = str(getattr(config, "accent_color", "#1DE9B6") or "#1DE9B6")
        self._drawing_brush_thickness = 8
        self._drawing_eraser_thickness = 18
        self._drawing_eraser_mode = "normal"
        self._camera_draw_canvas: np.ndarray | None = None
        self._camera_draw_history: list[tuple[np.ndarray, list[dict], bool]] = []
        self._camera_draw_strokes: list[dict] = []
        self._camera_draw_active_stroke_points: list[tuple[float, float]] = []
        self._camera_draw_raster_dirty = False
        self._camera_draw_last_point: tuple[int, int] | None = None
        self._camera_draw_erasing = False
        self._drawing_grabbed_stroke_index: int | None = None
        self._drawing_grab_last_point: tuple[int, int] | None = None
        self._drawing_grab_history_pushed = False
        self._drawing_secondary_hand_reading = None
        self._drawing_stretch_active = False
        self._drawing_stretch_initial_distance: float | None = None
        self._drawing_stretch_initial_points: list[tuple[float, float]] | None = None
        self._drawing_stretch_centroid: tuple[float, float] | None = None
        self._drawing_stretch_history_pushed = False
        self._drawing_request_token = 0
        self._drawing_swipe_cooldown_until = 0.0
        self._drawing_request_action = ""
        self._drawing_shape_mode = False
        self._drawing_draw_grace_until = 0.0
        self._drawing_erase_grace_until = 0.0
        self._drawing_thumb_open_streak = 0  # legacy counter, no longer used
        # Time-based pen-lift trigger. Stamped to monotonic_now the
        # first frame the thumb reads as open; reset to 0 the first
        # frame it doesn't. When the elapsed time exceeds
        # _DRAWING_THUMB_OPEN_HOLD_SECONDS, the pen lifts. Time-based
        # (vs frame-counted) so the hold duration is deterministic
        # across variable fps.
        self._drawing_thumb_open_since: float = 0.0
        # True for every frame the thumb-open lift signal is firing
        # right now (sustained or just-started). Set by the tool-
        # decision pass and consumed by the stroke renderer to skip
        # point appending while the user is lifting — without this,
        # the natural rightward hand motion during the 0.20 s lift
        # hold would drift the stroke endpoint. Brief one-frame
        # thumb_open misreads still recover (next frame this flips
        # back to False and the grace window keeps drawing).
        self._drawing_freeze_pen: bool = False
        # Anti-misfire: a new stroke only starts after 2 consecutive
        # draw-pose frames. Single-frame jitter (e.g. dropping the
        # hand from an open position briefly reads as draw pose for
        # one frame) doesn't extend grace and so doesn't paint a
        # stray dot. An already-active stroke (already in grace
        # window) ignores this counter and extends per-frame as
        # normal so tilt/rotation wobble doesn't break the stroke.
        self._drawing_draw_active_streak = 0
        self._drawing_wheel_candidate = "neutral"
        self._drawing_wheel_candidate_since = 0.0
        self._drawing_wheel_visible = False
        self._drawing_wheel_anchor = None
        self._drawing_wheel_selected_key: str | None = None
        self._drawing_wheel_selected_since = 0.0
        self._drawing_wheel_pose_grace_until = 0.0
        self._drawing_wheel_cooldown_until = 0.0
        self._drawing_wheel_cursor_offset: tuple[float, float] | None = None
        self.drawing_wheel_overlay = SpotifyWheelOverlay(config)
        self._utility_wheel_candidate = "neutral"
        self._utility_wheel_candidate_since = 0.0
        self._utility_wheel_visible = False
        self._utility_wheel_anchor = None
        self._utility_wheel_selected_key: str | None = None
        self._utility_wheel_selected_since = 0.0
        self._utility_wheel_pose_grace_until = 0.0
        self._utility_wheel_cooldown_until = 0.0
        self._utility_wheel_cursor_offset: tuple[float, float] | None = None
        self.utility_wheel_overlay = SpotifyWheelOverlay(config)
        self._utility_request_token = 0
        self._utility_request_action = ""
        self._utility_recording_active = False
        self._utility_recording_stop_candidate_since = 0.0
        self._utility_capture_selection_active = False
        self._utility_capture_cursor_norm = None
        self._utility_capture_left_down = False
        self._utility_capture_right_down = False
        self._utility_capture_clicks_armed = False
        self._utility_capture_clicks_armed = False
        self._utility_capture_clicks_armed = False
        self._utility_capture_selection_active = False
        self._utility_capture_cursor_norm: tuple[float, float] | None = None
        self._utility_capture_left_down = False
        self._utility_capture_right_down = False
        # Effective (shrunk) mouse-control-box bounds applied while a
        # hand-selector dialog is active. Used by the live-view
        # widget to render the red box during chooser sessions even
        # when full mouse mode is off.
        self._utility_capture_box_bounds: tuple[float, float, float, float] | None = None
        self._window_expand_candidate_since = 0.0
        self._window_contract_candidate_since = 0.0
        self._window_close_candidate_since = 0.0
        self._window_gesture_cooldown_until = 0.0
        self._window_pair_smoothed_distance: float | None = None
        self._window_pair_last_seen_at = 0.0
        self._window_pair_overlay = None
        self._window_sequence_start_state: str | None = None
        self._window_sequence_start_candidate: str | None = None
        self._window_sequence_start_candidate_since = 0.0
        self._window_sequence_target_candidate: str | None = None
        self._window_sequence_target_candidate_since = 0.0
        self._gestures_enabled = True
        self._selection_prompt_active = False
        self._selection_prompt_title = "Which file/folder?"
        self._selection_prompt_items: list[tuple[str, str, str]] = []
        self._selection_prompt_instruction = "Say the corresponding letter."
        self._save_prompt_active = False
        self._save_prompt_text = "Where would you like to save this file?"

        # Fallback timer: re-kicks the loop if a tick exits early
        # (cap.read failure, engine shutdown race) and never gets
        # rescheduled by the result-driven path. The hot path (and
        # the one that actually paces the gesture loop) is the
        # singleShot scheduled at the end of every _on_engine_result.
        # That path lets the next tick fire as soon as the previous
        # cycle ends, instead of waiting for a 15-ms-aligned timer
        # event â€” which we were observing get missed whenever a
        # cycle's work ran past its slot, capping FPS at 27 even
        # when per-cycle work measured under 12 ms.
        self._timer = QTimer(self)
        self._timer.setTimerType(Qt.PreciseTimer)
        self._timer.setInterval(15)
        self._timer.timeout.connect(self._tick)

        # Background-thread engine runner. The main-thread `_tick`
        # dispatches a frame here and returns immediately; inference
        # runs in parallel, and `_engine_result_ready` fires the
        # post-engine handler back on the main thread when done. See
        # `_EngineRunner` docstring for the full rationale â€” short
        # version: per-frame inference is the single heaviest CPU
        # step (12-25 ms MP / 2-7 ms GPU) and was the dominant chunk
        # of main-thread work, capping `actual self._fps` at 16-26
        # even with the GPU port. With this off-main, the QTimer can
        # fire as fast as the camera feeds frames.
        self._engine_runner = _EngineRunner()
        self._engine_runner._post_infer = self._attach_custom_hands_on_engine_thread
        self._engine_result_ready.connect(self._on_engine_result)

        # Custom-gesture live runner â€” owns its own classifier + hold/
        # cooldown state and fires actions when the user holds a
        # registered custom pose. Initializes from disk so any gesture
        # the user previously saved is live immediately.
        try:
            from ...custom_gestures.runner import CustomGestureRunner
            self._custom_gesture_runner = CustomGestureRunner(
                binding_resolver=self._custom_runner_binding_resolver,
                image_overlay_handler=self._custom_runner_image_overlay_handler,
            )
        except Exception as exc:
            print(f"[custom-gestures] runner init failed: {exc}")
            self._custom_gesture_runner = None
        # Rate-limit the GPU/Lite private MediaPipe pass used when
        # engine landmarks are not the recorder distribution. Runs on
        # the engine thread (see _attach_custom_hands_on_engine_thread),
        # every other inference, so the GUI camera tick stays free.
        self._custom_runner_slow_path_skip_ratio = 2
        # Start at 1 so the first GPU/Lite tick samples immediately
        # ((1+1)%2==0) instead of skipping with an empty cache.
        self._custom_runner_slow_path_counter = 1
        self._last_custom_mp_hands: list | None = None

        # Dynamic custom-gesture runtime. Parallel to the static
        # runner above — same registry, same `fire_once` cooldown
        # plumbing, but uses motion-based DTW matching instead of
        # pose-similarity. Loads on construction; auto-reloads on
        # registry-file mtime changes via maybe_reload_if_changed
        # alongside the static runner each tick.
        try:
            from ...custom_gestures.dynamic_runtime import DynamicGestureRuntime
            self._dynamic_gesture_runtime = DynamicGestureRuntime()
            self._dynamic_gesture_runtime.reload()
        except Exception as exc:
            print(f"[custom-gestures] dynamic runtime init failed: {exc}")
            self._dynamic_gesture_runtime = None
        try:
            from ...custom_gestures.pose_sequence_runtime import PoseSequenceRuntime
            self._pose_sequence_runtime = PoseSequenceRuntime()
            self._pose_sequence_runtime.reload()
        except Exception as exc:
            print(f"[custom-gestures] pose sequence runtime init failed: {exc}")
            self._pose_sequence_runtime = None
        self._seq_suppress_dynamic_until = 0.0
        # When a custom horizontal swipe owns the live pose, builtin
        # swipe_left/right must not also dispatch (sandbox never had
        # this collision because it has no builtin swipe path).
        self._suppress_builtin_horizontal_swipe = False
        self._custom_dyn_builtin_suppress_until = 0.0

        # Per-tick timing markers shared between _tick and
        # _on_engine_result â€” needed because the engine call now
        # returns asynchronously, so the post-engine handler can't
        # see _tick's local `t0`, `t_read`, `t_prep`. Format:
        # (debug_timing_enabled, t0, t_read, t_prep).
        self._tick_timing_state: tuple[bool, float, float, float] | None = None
        # Set True once a frame is submitted to the runner; cleared
        # when the queued result signal lands in _on_engine_result.
        # Together with the runner's `busy` flag this prevents a
        # second _tick from firing between "runner clears busy" and
        # "main thread handles the previous result", which would let
        # tick N+1 see stale `_last_result_had_hand` and make a wrong
        # skip-frame decision.
        self._async_result_pending: bool = False

        # C14 diagnostic: raw_frame_ready emit rate + tick cadence.
        # Logged every ~2 s so we can tell "the camera reader is
        # only delivering 25 fps" apart from "the tick loop is only
        # running 25 fps despite the reader delivering more." The
        # per-tick timing block already shows engine=Xms; this fills
        # in the tick-to-tick cadence view (paint→paint is a widget
        # metric, this is an engine-side metric).
        self._raw_emit_count: int = 0
        self._raw_emit_last_log: float = 0.0
        self._tick_call_count: int = 0

        # C22 diagnostic: system-level CPU / GPU / RAM every ~5 s so
        # the user can correlate mode toggles with real resource usage
        # from the app log alone (instead of having Task Manager open
        # side by side). NVIDIA GPU stats via nvidia-smi if it's on
        # PATH; skipped silently on non-NVIDIA or when the CLI is
        # missing. psutil-based process CPU + RSS is always emitted.
        # State kept minimal: just the last-log timestamp; everything
        # else is queried on demand.
        self._perf_monitor_last_log: float = 0.0
        try:
            import psutil as _psutil_mod
            self._perf_monitor_psutil = _psutil_mod
            self._perf_monitor_proc = _psutil_mod.Process()
            # Prime cpu_percent() so the first real call returns a
            # meaningful delta rather than 0. Same trick psutil docs
            # recommend for per-process cpu_percent.
            try:
                self._perf_monitor_proc.cpu_percent(interval=None)
            except Exception:
                pass
        except Exception:
            self._perf_monitor_psutil = None
            self._perf_monitor_proc = None

        # Per-frame timing samples used by the Lite Mode diagnostic
        # in _tick. Empty when Lite Mode is off; sampled at every
        # tick when on, summarised to stderr every 2s. Helps tell
        # camera-bound from CPU-bound frames apart when an FPS
        # report comes in.
        self._timing_samples: deque[tuple[float, float, float, float, float, float, float, float]] = deque(maxlen=240)
        self._last_timing_log: float = 0.0

        # Skip-frame inference state. Empty frames skip MediaPipe on
        # a cadence (every other tick, then 2-of-3 after a short idle)
        # because the detector is the single biggest CPU cost
        # (12-25 ms) and there's nothing it could surface on an empty
        # frame that the next inference tick won't catch. As soon as a
        # hand appears we go back to full-rate inference so dynamic
        # gestures (swipe, repeat-circle) — which depend on
        # frame-by-frame motion — are never sampled at half-rate.
        self._last_result_had_hand: bool = False
        self._inference_skipped_last: bool = False
        self._idle_skip_ticks: int = 0
        # C13 (v1.1.7): track whether the last engine_landmarks_ready
        # emit carried a non-empty hands list. When no hand is detected
        # in the current cycle AND the previous emit also had no hands
        # AND there's no mouse-mode overlay + no drawing-mode overlay
        # override, the payload is byte-for-byte identical to the last
        # one and just re-runs both live-view widgets' slot handlers
        # for no visible change. Eliding the redundant emit lets the
        # widgets skip an update_landmarks + paint cycle each. The
        # hand-present-to-absent transition still fires exactly one
        # final neutral emit so the widget clears its stashed skeleton
        # — that's what prevents "hand leaves the frame but the
        # skeleton stays" ghosting.
        self._last_emit_had_hands: bool = False

        # Wall-clock-rate-limited debug-frame emit. The receivers
        # (mini viewer + live view) each do cv2.cvtColor + QImage +
        # QPixmap.fromImage + scaled-with-smooth-transformation per
        # emit, totalling 15-30 ms of main-thread work. When the
        # gesture loop produces emits at 60-70 Hz but receivers can
        # only render at 25-30 Hz, the Qt cross-thread queued-signal
        # queue accumulates frames at ~30/second and the user sees
        # 1-2 seconds of perceived display lag on top of the actual
        # capture latency. Rate-limiting emits by wall-clock time
        # guarantees we never push more frames than the receivers
        # can render, so the queue never grows. 30 Hz is plenty for
        # smooth-looking live preview; gesture detection still runs
        # at full loop rate underneath. `_action_history_dirty_
        # for_emit` flips True when a new action gets recorded so
        # the next emit fires regardless of the rate-limit cooldown,
        # keeping toasts / overlays punctual.
        # 30 Hz emit cap. Earlier we throttled to 20 Hz because the
        # receivers' cvtColor + QImage + QPixmap.scaled(Smooth)
        # pipeline cost ~25-50 ms per emit at 720p across mini +
        # live view, and 30 Hz let the Qt cross-thread queue grow
        # during busy stretches. Two changes since make 30 Hz safe:
        # (1) hidden receivers short-circuit before any conversion
        # work, so closed previews cost nothing; (2) the visible
        # receivers now resize via cv2.INTER_AREA on the source
        # BGR before cvtColor, dropping per-emit cost from ~6 ms
        # to <1 ms. 30 Hz gives noticeably less perceived motion
        # smear without re-introducing the queue-growth lag.
        self._emit_min_interval_seconds: float = 1.0 / 30.0
        self._last_emit_monotonic: float = 0.0
        self._action_history_dirty_for_emit: bool = False

    def reload_custom_gestures(self) -> None:
        """Re-read custom gestures from disk. Called by the Settings
        panel after the user adds / edits / deletes a gesture so the
        live pipeline picks up the change without restarting the app."""
        runner = getattr(self, "_custom_gesture_runner", None)
        if runner is None:
            try:
                from ...custom_gestures.runner import CustomGestureRunner
                self._custom_gesture_runner = CustomGestureRunner(
                    binding_resolver=self._custom_runner_binding_resolver,
                    image_overlay_handler=self._custom_runner_image_overlay_handler,
                )
            except Exception as exc:
                print(f"[custom-gestures] late init failed: {exc}")
            return
        try:
            runner.reload()
        except Exception as exc:
            print(f"[custom-gestures] reload failed: {exc}")
        dyn = getattr(self, "_dynamic_gesture_runtime", None)
        if dyn is not None:
            try:
                dyn.reload()
            except Exception as exc:
                print(f"[custom-gestures] dynamic reload failed: {exc}")
        seq = getattr(self, "_pose_sequence_runtime", None)
        if seq is not None:
            try:
                seq.reload()
            except Exception as exc:
                print(f"[custom-gestures] pose sequence reload failed: {exc}")

    def _emit_voice_log_step(self, message: str) -> None:
        """Forward a step message from VoiceCommandProcessor to the
        user-facing detailed log. Wrapped in try/except because the
        emit can theoretically fire from a worker thread; we don't
        want a Qt-side error to bubble up and crash the running
        voice command."""
        try:
            self.engine_log.emit(str(message))
        except Exception:
            pass

    def _record_action(self, label: str, display_text: str) -> None:
        if not label or label == "-":
            return
        # Used to skip every *_failed / *_idle / *_requires_open /
        # *_closed event silently. That meant the user couldn't see
        # WHY a command didn't fire — the action history showed only
        # the successes. Now we record them so the action log has the
        # full picture, including the failure variants. The undo
        # logic still ignores non-undoable labels via the
        # _UNDO_LABEL_PAIRS lookup below, so the user can't accidentally
        # "undo" a failure entry.
        undoable = label in _UNDO_LABEL_PAIRS
        event = ActionEvent(
            timestamp=time.time(),
            label=label,
            display_text=display_text or label,
            undoable=undoable,
        )
        with self._action_history_lock:
            self._action_history.append(event)
            snapshot = list(self._action_history)
        # Promote the next throttled-emit to fire immediately so
        # the user sees the toast / overlay update without the
        # 1-frame skew the throttle would otherwise impose.
        self._action_history_dirty_for_emit = True
        try:
            self.action_history_changed.emit(snapshot)
        except Exception:
            pass
        # Telemetry: anonymous "user did something" event. Mirrors
        # the in-app action history exactly — every record_action
        # call is a user-visible action regardless of whether it
        # came through _dispatch_action, the volume tracker, swipe
        # router, wheels, voice, or drawing pipeline. The shared
        # `_action_fired_telemetry_last` dict dedupes against the
        # tail of _dispatch_action's `if fired:` block so flows that
        # touch both (voice_command_listen → _start_voice_command
        # → _record_action) only emit once.
        try:
            last_telemetry = getattr(self, "_action_fired_telemetry_last", None)
            if last_telemetry is None:
                last_telemetry = {}
                self._action_fired_telemetry_last = last_telemetry
            now = time.monotonic()
            if now - last_telemetry.get(label, 0.0) >= 0.25:
                last_telemetry[label] = now
                from ...telemetry import track as _track
                # Derive an `app` bucket from the action_id prefix so the
                # dashboard can group by surface — chrome_search /
                # chrome_back / chrome_forward all roll up under
                # app="chrome", youtube_next / youtube_toggle under
                # app="youtube", spotify_* under app="spotify", etc.
                # Without this, the dashboard's only handle on what the
                # user is doing is the raw action_id string, which is
                # too noisy to chart over.
                app_bucket = "other"
                label_lower = str(label).lower()
                for prefix in (
                    "chrome", "youtube", "spotify", "drawing", "mouse",
                    "voice", "volume", "dictation", "screenshot",
                    "recording", "clip", "system", "window",
                ):
                    if label_lower.startswith(prefix + "_") or label_lower == prefix:
                        app_bucket = prefix
                        break
                _track(
                    "action_fired",
                    {
                        "action_id": str(label),
                        "app": app_bucket,
                        "in_tutorial": bool(getattr(self, "_tutorial_mode_enabled", False)),
                    },
                )
        except Exception:
            pass

    def undo_last_action(self) -> bool:
        with self._action_history_lock:
            target: ActionEvent | None = None
            for event in reversed(self._action_history):
                if event.is_undo:
                    continue
                if event.undoable:
                    target = event
                    break
        if target is None:
            return False
        inverse_label = _UNDO_LABEL_PAIRS.get(target.label)
        if inverse_label is None:
            return False
        performed = self._perform_action_by_label(inverse_label)
        if not performed:
            return False
        event = ActionEvent(
            timestamp=time.time(),
            label=inverse_label,
            display_text=f"undo: {target.display_text}",
            undoable=False,
            is_undo=True,
        )
        with self._action_history_lock:
            self._action_history.append(event)
            snapshot = list(self._action_history)
        try:
            self.action_history_changed.emit(snapshot)
        except Exception:
            pass
        return True

    def _perform_action_by_label(self, label: str) -> bool:
        try:
            if label == "spotify_next":
                return bool(self.spotify_controller.next_track())
            if label == "spotify_previous":
                return bool(self.spotify_controller.previous_track())
            if label == "spotify_toggle":
                return bool(self.spotify_controller.toggle_playback())
            if label == "spotify_shuffle":
                return bool(self.spotify_controller.toggle_shuffle())
            if label == "spotify_repeat":
                return bool(self.spotify_controller.toggle_repeat_track())
            if label == "youtube_next":
                return bool(self.youtube_controller.next_track())
            if label == "youtube_previous":
                return bool(self.youtube_controller.previous_track())
            if label == "youtube_toggle":
                return bool(self.youtube_controller.toggle_playback())
            if label == "chrome_back":
                return bool(self.chrome_controller.navigate_back())
            if label == "chrome_forward":
                return bool(self.chrome_controller.navigate_forward())
        except Exception:
            return False
        return False

    def action_history_snapshot(self) -> list[ActionEvent]:
        with self._action_history_lock:
            return list(self._action_history)

    def _prime_voice_app_hints_async(self) -> None:
        if self._app_hint_thread is not None and self._app_hint_thread.is_alive():
            return

        def _worker() -> None:
            try:
                self.voice_listener.set_app_hints(self.voice_processor.desktop_controller.application_hint_names())
            except Exception:
                pass

        self._app_hint_thread = threading.Thread(
            target=_worker,
            name="hgr-app-hints",
            daemon=True,
        )
        self._app_hint_thread.start()


    def _prime_voice_runtime_async(self) -> None:
        def _worker() -> None:
            try:
                self.voice_listener.prewarm()
            except Exception:
                pass

        threading.Thread(
            target=_worker,
            name="hgr-voice-prewarm",
            daemon=True,
        ).start()

    def _dictation_backend_label(self) -> str:
        def _pretty(name: str | None) -> str:
            if not name:
                return ""
            lower = name.lower()
            if lower == "cuda":
                return "CUDA"
            if lower == "vulkan":
                return "Vulkan"
            if lower == "cpu":
                return "CPU"
            return name.upper() if len(name) <= 4 else name.capitalize()

        dict_backend = _pretty(getattr(self.live_dictation_streamer, "backend", None))
        gram_backend = _pretty(getattr(self.llama_server, "backend", None)) if self.llama_server.available else ""
        if dict_backend and gram_backend and dict_backend != gram_backend:
            return f"Using {dict_backend} / {gram_backend}"
        if dict_backend:
            return f"Using {dict_backend}"
        if gram_backend:
            return f"Using {gram_backend}"
        return ""

    def _apply_grammar_correction(self, result: CorrectionResult) -> None:
        original = str(result.original or "")
        corrected = str(result.corrected or "")
        if not original or not corrected or original == corrected:
            return
        # Sanity guard against llama hallucinations / paraphrasing.
        # Allow legitimate de-duplication (corrections that strip stuttered
        # repeats) â€” those can drop ~half the chars/words and still be right.
        # Reject only if the result is obviously broken: empty, drastically
        # truncated, or wildly expanded.
        if not corrected.strip("\n\r\t .,;:!?'\""):
            print(f"[grammar] rejecting correction: empty/punctuation-only")
            return
        orig_len = len(original)
        corr_len = len(corrected)
        ratio = corr_len / orig_len if orig_len else 1.0
        orig_words = len([w for w in original.split() if w.strip()])
        corr_words = len([w for w in corrected.split() if w.strip()])
        word_delta = corr_words - orig_words
        if ratio < 0.4 or ratio > 1.6 or word_delta > 6:
            print(f"[grammar] rejecting correction: ratio={ratio:.2f} word_delta={word_delta:+d} (orig={orig_len}c/{orig_words}w corr={corr_len}c/{corr_words}w)")
            return
        with self._corrector_lock:
            if self.grammar_corrector.is_chunk_stale():
                print(f"[grammar] skipping apply: chunk stale after lock")
                return
            # Don't apply a correction while live hypotheses are mid-utterance:
            # the diff-based replace_text assumes the dictated tail on screen is
            # stable, but a hypothesis could be appending to it concurrently. The
            # corrector fires on ~10s idle so this is rare, but the guard makes it
            # safe (the next idle pass re-corrects once typing has settled).
            state = self._dictation_state
            if state is not None and (
                state.get("committed")
                or state.get("stream_hyp_chars")
                or state.get("pending_final_text")
            ):
                print(f"[grammar] skipping apply: live typing in progress")
                return
            tail = self.grammar_corrector.snapshot_tail()
            previous = original + tail
            replacement = corrected + tail
            tic = self.text_input_controller
            try:
                target_hwnd = int(getattr(tic, "_target_hwnd", 0) or 0)
                foreground_hwnd = tic._foreground_window() if hasattr(tic, "_foreground_window") else 0
            except Exception:
                target_hwnd = 0
                foreground_hwnd = 0
            focus_ok = target_hwnd > 0 and target_hwnd == foreground_hwnd
            try:
                success = tic.replace_text(previous, replacement)
            except Exception as exc:
                print(f"[grammar] replace_text raised: {exc}")
                success = False
            if success:
                self._corrector_applied_message = "dictation corrected"
                print(f"[grammar] applied correction (-{len(previous)} +{len(replacement)} chars) focus_ok={focus_ok} target={target_hwnd} fg={foreground_hwnd}")
            else:
                self._corrector_applied_message = "correction skipped (focus lost)"
                print(f"[grammar] apply failed focus_ok={focus_ok} target={target_hwnd} fg={foreground_hwnd} msg={tic.message}")

    def _apply_refinement(self, result: RefinementResult) -> None:
        if not self._dictation_active:
            return
        refined_text = (result.text or "").strip()
        print(f"[refiner] received: {refined_text!r} ({result.duration_seconds:.2f}s)")
        if not refined_text:
            return
        refined_text = _strip_whisper_hallucinations(refined_text)
        if not refined_text:
            return
        with self._corrector_lock:
            state = self._dictation_state
            if state is None:
                return
            prev_text = str(state.get("last_final_text", "") or "")
            if not prev_text or state.get("last_final_refined"):
                return
            if state.get("committed"):
                # streaming hypothesis already typing the next utterance â€” skip
                return
            if state.get("stream_hyp_chars", 0):
                return
            typed_len = int(state.get("last_final_typed_len", 0))
            if typed_len <= 0 or typed_len > 600:
                return
            age = time.monotonic() - float(state.get("last_final_time", 0.0))
            if age > 8.0:
                return
            if refined_text == prev_text:
                state["last_final_refined"] = True
                return
            # safety: don't apply if the refined text is wildly larger than what
            # was streamed â€” usually means VAD captured multiple utterances and
            # we'd duplicate text in the editor.
            if len(refined_text) > max(60, int(len(prev_text) * 2.0)):
                print(f"[refiner] skipping: refined too large ({len(refined_text)} vs streamed {len(prev_text)})")
                state["last_final_refined"] = True
                return
            to_type = refined_text + " "
            tic = self.text_input_controller
            try:
                removed = tic.remove_text(typed_len)
            except Exception as exc:
                print(f"[refiner] remove_text raised: {exc}")
                return
            if not removed:
                return
            try:
                inserted = tic.insert_text(to_type)
            except Exception as exc:
                print(f"[refiner] insert_text raised: {exc}")
                return
            if not inserted:
                return
            try:
                self.grammar_corrector.sync_replace(typed_len, to_type)
            except Exception as exc:
                print(f"[refiner] sync_replace raised: {exc}")
            current_display = str(state.get("final_display", "") or "")
            if current_display.endswith(prev_text):
                current_display = current_display[: len(current_display) - len(prev_text)] + refined_text
            else:
                current_display = (current_display + " " + refined_text).strip() if current_display else refined_text
            state["final_display"] = current_display
            state["last_final_text"] = refined_text
            state["last_final_typed_len"] = len(to_type)
            state["last_final_refined"] = True
            print(f"[refiner] applied refinement: -{typed_len} +{len(to_type)} chars")

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def gestures_enabled(self) -> bool:
        return bool(self._gestures_enabled)

    def set_gestures_enabled(self, enabled: bool) -> None:
        enabled = bool(enabled)
        if enabled == self._gestures_enabled:
            return
        self._gestures_enabled = enabled
        if not enabled:
            self.mouse_controller.release_all()
            self.mouse_tracker.reset()
            self._last_mouse_update = self._blank_mouse_update()
            self._mouse_mode_enabled = False
            self._mouse_control_text = "Gestures disabled"
            self._mouse_status_text = "off"
            self._reset_chrome_wheel(clear_cooldown=False)
            self._reset_spotify_wheel(clear_cooldown=False)
            self._reset_youtube_wheel(clear_cooldown=False)
            self._reset_drawing_wheel(clear_cooldown=False)
            self._reset_utility_wheel(clear_cooldown=False)
            self._reset_window_gesture_state(clear_cooldown=False)
            self._drawing_cursor_norm = None
            self._drawing_tool = "hidden"
            self._camera_draw_last_point = None
            self._window_pair_smoothed_distance = None
            self._window_pair_overlay = None
            self._chrome_control_text = "Gestures disabled"
            self._spotify_control_text = "Gestures disabled"
            self._volume_mode_active = False
            self._volume_status_text = "paused"
            self._volume_overlay_visible = False
            self._update_volume_overlay()
        else:
            self._chrome_control_text = self.chrome_controller.message
            self._spotify_control_text = self.spotify_controller.message
            if self.mouse_controller.available:
                self._mouse_control_text = "Mouse mode off"
                self._mouse_status_text = "off"

    def _reset_drawing_runtime(self, *, keep_mode: bool = False) -> None:
        if not keep_mode:
            self._drawing_mode_enabled = False
        self._drawing_toggle_candidate_since = 0.0
        self._drawing_cursor_norm = None
        self._drawing_tool = "hidden"
        self._drawing_swipe_cooldown_until = 0.0
        self._drawing_grabbed_stroke_index = None
        self._drawing_grab_last_point = None
        self._drawing_grab_history_pushed = False
        self._drawing_stretch_active = False
        self._drawing_stretch_initial_distance = None
        self._drawing_stretch_initial_points = None
        self._drawing_stretch_centroid = None
        self._drawing_stretch_history_pushed = False
        # Shape mode is a per-session toggle: it doesn't survive
        # leaving drawing mode (so re-entering drawing mode always
        # starts on plain freehand, the most common state) and it
        # doesn't survive an app restart (the field is re-initialized
        # to False in __init__; nothing persists it to config). The
        # tools that DO persist â€” color, thickness, eraser kind â€”
        # are owned by AppConfig, not the engine runtime, so they
        # are unaffected by this reset.
        if self._drawing_shape_mode:
            self._drawing_shape_mode = False
            self._queue_drawing_request("shape_off")
        self._drawing_control_text = "Drawing mode on" if self._drawing_mode_enabled else "Drawing mode off"

    def _toggle_drawing_mode(self, now: float) -> None:
        self._drawing_mode_enabled = not self._drawing_mode_enabled
        self._drawing_toggle_candidate_since = 0.0
        self._drawing_toggle_cooldown_until = now + 1.0
        self._drawing_cursor_norm = None
        self._drawing_tool = "hidden"
        self._camera_draw_last_point = None
        # Shape mode is per-session: any time we leave drawing mode
        # we clear it so re-entering always starts on plain freehand.
        # It also can never survive an app restart (the field is
        # initialised to False in __init__ and not persisted to
        # config). Color, thickness, and eraser kind ARE persisted
        # via AppConfig â€” they're not touched here.
        if not self._drawing_mode_enabled and self._drawing_shape_mode:
            self._drawing_shape_mode = False
            self._queue_drawing_request("shape_off")
        self._reset_drawing_wheel(clear_cooldown=False)
        self._reset_chrome_wheel(clear_cooldown=False)
        self._reset_spotify_wheel(clear_cooldown=False)
        self._reset_window_gesture_state(clear_cooldown=False)
        self.chrome_router.reset()
        self.spotify_router.reset()
        self.youtube_router.reset()
        self.mouse_controller.release_all()
        self.mouse_tracker.reset()
        self._last_mouse_update = self._blank_mouse_update()
        self._mouse_mode_enabled = False
        self._mouse_status_text = "off" if self.mouse_controller.available else "unavailable"
        self._mouse_control_text = "Mouse mode off" if self.mouse_controller.available else self.mouse_controller.message
        self._volume_mode_active = False
        self._volume_status_text = "paused"
        self._volume_overlay_visible = False
        self._update_volume_overlay()
        self.voice_status_overlay.hide_overlay()
        state = "enabled" if self._drawing_mode_enabled else "disabled"
        self._drawing_control_text = f"Drawing mode {state}"
        try:
            self.voice_status_overlay.show_info_hint(
                "Draw mode: ON" if self._drawing_mode_enabled else "Draw mode: OFF",
                duration=3.0,
            )
        except Exception:
            pass
        try:
            self.command_detected.emit(self._drawing_control_text)
        except Exception:
            pass
        # Log to recent-actions so gesture-driven drawing toggles show
        # up in the action history (was previously emit-only).
        self._record_action(
            "drawing_mode_on" if self._drawing_mode_enabled else "drawing_mode_off",
            "Drawing mode on" if self._drawing_mode_enabled else "Drawing mode off",
        )

    def _handle_drawing_toggle(self, prediction, hand_handedness: str | None, now: float) -> bool:
        left_pred = prediction if hand_handedness == "Left" else self._left_hand_prediction
        if left_pred is None:
            self._drawing_toggle_candidate_since = 0.0
            return False
        stable_label = str(getattr(left_pred, "stable_label", "neutral") or "neutral")
        if stable_label != "four":
            self._drawing_toggle_candidate_since = 0.0
            return False
        if now < self._drawing_toggle_cooldown_until:
            return True
        if self._drawing_toggle_candidate_since <= 0.0:
            self._drawing_toggle_candidate_since = now
            self._drawing_control_text = "hold left hand four for drawing mode"
            return True
        if now - self._drawing_toggle_candidate_since >= _STATIC_GESTURE_HOLD_SECONDS:
            self._toggle_drawing_mode(now)
            return True
        return True

    def _drawing_finger_extendedish(self, finger, *, primary: bool = False) -> bool:
        if finger is None:
            return False
        if finger.state in {"fully_open", "partially_curled"}:
            return True
        threshold = 0.50 if primary else 0.56
        return float(getattr(finger, "openness", 0.0) or 0.0) >= threshold

    def _drawing_finger_foldedish(self, finger) -> bool:
        if finger is None:
            return False
        if finger.state in {"closed", "mostly_curled"}:
            return True
        openness = float(getattr(finger, "openness", 0.0) or 0.0)
        curl = float(getattr(finger, "curl", 0.0) or 0.0)
        return openness <= 0.48 or curl >= 0.48

    def _drawing_thumb_foldedish(self, finger) -> bool:
        if finger is None:
            return False
        if finger.state in {"closed", "mostly_curled", "partially_curled"}:
            return True
        # Rotation-invariant tucked-thumb detection. MediaPipe's
        # `openness` and `state` are derived from the wristâ†’tip
        # extension distance, which grows when the hand tilts even
        # slightly â€” so a thumb that's still physically tucked
        # against the palm starts reading as 'fully_open' the
        # moment the user rotates their wrist. Two more stable
        # signals exist on the finger object: `palm_distance`
        # (tip-to-palm-center, smaller when tucked) and the bend
        # angles (low when joints are folded). Either of those
        # confirming a tucked posture is enough to treat the thumb
        # as folded for the drawing path.
        palm_distance = float(getattr(finger, "palm_distance", 1.0) or 1.0)
        curl = float(getattr(finger, "curl", 0.0) or 0.0)
        bend_distal = float(getattr(finger, "bend_distal", 180.0) or 180.0)
        bend_proximal = float(getattr(finger, "bend_proximal", 180.0) or 180.0)
        openness = float(getattr(finger, "openness", 0.0) or 0.0)
        return (
            palm_distance < 0.55
            or curl >= 0.10
            or bend_distal < 150.0
            or bend_proximal < 140.0
            or openness <= 0.92
        )

    def _drawing_draw_pose_active(self, prediction, hand_reading) -> bool:
        if hand_reading is None or prediction is None:
            return False
        fingers = hand_reading.fingers
        index = fingers.get("index")
        middle = fingers.get("middle")
        ring = fingers.get("ring")
        pinky = fingers.get("pinky")
        if index is None or middle is None or ring is None or pinky is None:
            return False

        # Base geometry â€” required by EITHER path below. Index has
        # to be extended-ish, the other three folded-ish, middle
        # not visibly spread away from the palm. These are loose
        # enough that any honest 'one' shape passes them, but
        # exclude clearly different poses (open hand, fist, two,
        # three) without depending on the thumb (which drifts
        # under wrist rotation and was the source of the previous
        # round's regressions).
        outer_folded = sum(
            1 for name in ("middle", "ring", "pinky")
            if self._drawing_finger_foldedish(fingers.get(name))
        )
        base_geometry = (
            self._drawing_finger_extendedish(index, primary=True)
            and outer_folded >= 2
            and float(getattr(middle, "openness", 0.0) or 0.0) <= 0.60
        )
        if not base_geometry:
            return False

        # Path 1 â€” recogniser confirms 'one'. Either via the
        # stabilised label or via a moderate-confidence raw
        # match (lowered the threshold to 0.40 because the user
        # reported pinch-shaped tightening that left 'one'
        # confidence borderline on real strokes). Fast-path; skips
        # the stricter geometry below.
        stable = str(getattr(prediction, "stable_label", "neutral") or "neutral")
        raw = str(getattr(prediction, "raw_label", "neutral") or "neutral")
        confidence = float(getattr(prediction, "confidence", 0.0) or 0.0)
        if stable == "one" or (raw == "one" and confidence >= 0.40):
            return True

        # Path 2 â€” recogniser hasn't labelled this as 'one' (could
        # be wrist rotation / occlusion / borderline confidence)
        # but the geometry is unambiguous. Tighter than Path 1's
        # base check so 'two' / 'three' / 'pinch' / 'mute' don't
        # leak through:
        #   - index must be `fully_open` (not partially_curled),
        #     which excludes pinch and mid-transition shapes;
        #   - openness on the index â‰¥ 0.65 â€” a real one;
        #   - middle / ring / pinky must each be in
        #     {closed, mostly_curled} â€” excludes any 'two' or
        #     'three' frame where one of them is still
        #     partially_curled;
        #   - middle openness â‰¤ 0.30 â€” even tighter than the
        #     base 0.60, so a recogniser glitch that briefly
        #     reads middle as half-open can't slip through.
        strict_one = (
            index.state == "fully_open"
            and float(getattr(index, "openness", 0.0) or 0.0) >= 0.65
            and middle.state in ("closed", "mostly_curled")
            and ring.state in ("closed", "mostly_curled")
            and pinky.state in ("closed", "mostly_curled")
            and float(getattr(middle, "openness", 0.0) or 0.0) <= 0.30
        )
        return strict_one

    def _drawing_erase_pose_active(self, prediction, hand_reading) -> bool:
        if hand_reading is None:
            return False
        fingers = hand_reading.fingers
        spread = hand_reading.spreads.get("index_middle")
        spread_ok = False
        if spread is not None:
            spread_ok = spread.state == "together" or float(spread.distance) <= 0.60 or float(spread.together_strength) >= 0.14
        thumb_folded = self._drawing_thumb_foldedish(fingers.get("thumb"))
        ring_folded = self._drawing_finger_foldedish(fingers.get("ring"))
        pinky_folded = self._drawing_finger_foldedish(fingers.get("pinky"))
        folded_support = sum(1 for ok in (thumb_folded, ring_folded, pinky_folded) if ok)
        return (
            self._drawing_finger_extendedish(fingers.get("index"), primary=True)
            and self._drawing_finger_extendedish(fingers.get("middle"), primary=True)
            and folded_support >= 2
            and spread_ok
        )

    def _drawing_lift_pose_active(self, hand_reading) -> bool:
        if hand_reading is None:
            return False
        fingers = hand_reading.fingers
        thumb = fingers.get("thumb")
        if thumb is None or thumb.state not in {"fully_open", "partially_curled"} or thumb.openness < 0.56:
            return False
        extended = sum(
            1
            for name in ("index", "middle", "ring", "pinky")
            if self._drawing_finger_extendedish(fingers.get(name))
        )
        return extended >= 3

    def _drawing_pinch_pose_active(self, hand_reading) -> bool:
        if hand_reading is None:
            return False
        spread = hand_reading.spreads.get("thumb_index")
        if spread is None:
            return False
        distance = float(spread.distance)
        together_strength = float(spread.together_strength)
        pinch_ok = (
            spread.state == "together"
            or distance <= 0.48
            or together_strength >= 0.22
        )
        tight_pinch = distance <= 0.30 or together_strength >= 0.40
        fingers = hand_reading.fingers
        index_curled = self._drawing_finger_foldedish(fingers.get("index"))
        middle_curled = self._drawing_finger_foldedish(fingers.get("middle"))
        ring_curled = self._drawing_finger_foldedish(fingers.get("ring"))
        pinky_curled = self._drawing_finger_foldedish(fingers.get("pinky"))
        thumb = fingers.get("thumb")
        thumb_open = False
        if thumb is not None:
            thumb_open = thumb.state in {"fully_open", "partially_curled"} and float(getattr(thumb, "openness", 0.0) or 0.0) >= 0.45
        claw_shape = thumb_open and index_curled and middle_curled and ring_curled and pinky_curled
        if claw_shape:
            return True
        if tight_pinch and thumb_open and index_curled:
            return True
        if not pinch_ok:
            return False
        index_extended = self._drawing_finger_extendedish(fingers.get("index"), primary=True)
        if index_extended:
            return False
        outer_folded = sum(1 for curled in (ring_curled, pinky_curled) if curled)
        return outer_folded >= 1

    def _drawing_wheel_pose_active(self, prediction) -> bool:
        if prediction is None:
            return False
        stable_label = str(getattr(prediction, "stable_label", "neutral") or "neutral")
        raw_label = str(getattr(prediction, "raw_label", "neutral") or "neutral")
        confidence = float(getattr(prediction, "confidence", 0.0) or 0.0)
        return stable_label == "wheel_pose" or (raw_label == "wheel_pose" and confidence >= 0.52)

    def _map_drawing_control_box(self, x: float, y: float) -> tuple[float, float]:
        """Map a normalized fingertip coord through the drawing control box.

        A small square region of the frame (size x size, centered on the
        configured point) fills the whole [0,1] canvas, so the finger only
        moves within a forearm-sized patch. Square in normalized coords
        keeps motion undistorted on a 16:9 frame/canvas.
        """
        # Default enlarged 0.45 -> 0.72 -> 0.90: a small patch mapped to
        # the whole canvas meant high gain (0.45=~2.2x, 0.60=~1.7x,
        # 0.72=~1.4x all still felt disconnected from the monitor —
        # small hand motion = big cursor jump). 0.90 (~1.11x) maps
        # the camera frame ~1:1 to the monitor with a 10% edge buffer
        # so the hand stays in frame at the corners; centered to match
        # the user's natural reach across the full monitor width.
        # Effective when the user is on the new defaults; explicit
        # tunings are preserved by the settings.json migration in
        # app_config (see `drawing_control_box_size` migration block).
        size = float(getattr(self.config, "drawing_control_box_size", 0.90))
        size = max(0.15, min(1.0, size))
        cx = float(getattr(self.config, "drawing_control_box_center_x", 0.50))
        cy = float(getattr(self.config, "drawing_control_box_center_y", 0.50))
        half = size * 0.5
        min_x = min(max(cx - half, 0.0), 1.0 - size)
        min_y = min(max(cy - half, 0.0), 1.0 - size)
        mapped_x = (x - min_x) / size
        mapped_y = (y - min_y) / size
        return (
            max(0.0, min(1.0, mapped_x)),
            max(0.0, min(1.0, mapped_y)),
        )

    def _update_drawing_controls(self, prediction, hand_reading, hand_handedness: str | None, now: float) -> None:
        if not self._drawing_mode_enabled:
            self._drawing_cursor_norm = None
            self._drawing_tool = "hidden"
            return
        if hand_handedness != "Right" or hand_reading is None:
            self._drawing_cursor_norm = None
            self._drawing_tool = "hidden"
            self._drawing_control_text = f"Drawing mode enabled ({self._drawing_render_target})"
            self._camera_draw_last_point = None
            return
        try:
            raw_x = max(0.0, min(1.0, float(hand_reading.landmarks[8][0])))
            raw_y = max(0.0, min(1.0, float(hand_reading.landmarks[8][1])))
        except Exception:
            self._drawing_cursor_norm = None
            self._drawing_tool = "hidden"
            self._camera_draw_last_point = None
            return
        # Drawing cursor pipeline (two stages, both fixing prior
        # complaints that the pen felt jittery and forced full-frame
        # arm sweeps):
        #   1. One Euro filter on the RAW fingertip — adaptive
        #      jitter/lag tradeoff: steady when held still, low lag on
        #      fast strokes. Replaces the old fixed velocity-adaptive
        #      EMA, which still passed ~30% of jitter at rest.
        #   2. Control-box remap — a small square patch of the frame
        #      fills the whole canvas (like air-mouse), so the finger
        #      moves within a forearm-sized area instead of sweeping
        #      the full frame.
        # Filtering happens BEFORE the remap so filter tuning is
        # independent of the box gain.
        if getattr(self, "_drawing_oef_x", None) is None:
            min_cutoff = self._DRAWING_OEF_MIN_CUTOFF
            beta = self._DRAWING_OEF_BETA
            oef_max_dt = None
            if sys.platform == "darwin":
                # Irregular Mac capture dt made OneEuro treat landmark
                # jitter as motion (cursor crawl). Heavier rest cutoff,
                # clamp dt, keep beta high enough for strokes.
                min_cutoff = 0.20
                beta = 0.50
                oef_max_dt = 0.045
            self._drawing_oef_x = OneEuroFilter(
                min_cutoff=min_cutoff, beta=beta
            )
            self._drawing_oef_y = OneEuroFilter(
                min_cutoff=min_cutoff, beta=beta
            )
            self._drawing_oef_max_dt = oef_max_dt
        # Re-seed the filters whenever the cursor was just lost (drawing
        # mode entered or hand re-acquired) so it doesn't crawl in from
        # a stale position.
        if self._drawing_cursor_norm is None:
            self._drawing_oef_x.reset()
            self._drawing_oef_y.reset()
        oef_max_dt = getattr(self, "_drawing_oef_max_dt", None)
        smooth_x = self._drawing_oef_x.update(raw_x, now, max_dt=oef_max_dt)
        smooth_y = self._drawing_oef_y.update(raw_y, now, max_dt=oef_max_dt)
        # CAMERA ("switch view") draws on the camera feed the user is looking
        # at, so the ink must land exactly under the fingertip: map the
        # (smoothed) raw fingertip 1:1 to the canvas, with NO control-box gain.
        # This also removes the "moves too fast" feel in that view — the cursor
        # now tracks the finger one-to-one. SCREEN view is an air-mouse onto the
        # whole desktop (the camera isn't visible), so it keeps the control-box
        # remap that lets a forearm-sized patch cover the screen.
        if self._drawing_render_target == "camera":
            mapped = (
                max(0.0, min(1.0, smooth_x)),
                max(0.0, min(1.0, smooth_y)),
            )
        else:
            mapped = self._map_drawing_control_box(smooth_x, smooth_y)
        if (
            sys.platform == "darwin"
            and self._drawing_cursor_norm is not None
        ):
            lx, ly = self._drawing_cursor_norm
            dx = mapped[0] - lx
            dy = mapped[1] - ly
            # ~0.35% of the canvas — hold-still tremor on FaceTime.
            if (dx * dx + dy * dy) < (0.0035 * 0.0035):
                mapped = (lx, ly)
        self._drawing_cursor_norm = mapped
        if self._drawing_lift_pose_active(hand_reading):
            self._drawing_tool = "hover"
            self._drawing_control_text = f"drawing hover ({self._drawing_render_target})"
            self._camera_draw_last_point = None
            return
        # Pen-lift trigger: open the thumb for >= 0.20 s and the
        # pen lifts. Detection requires multiple corroborating
        # signals so a rotated single-finger pose (where the
        # state classifier sometimes mis-labels the thumb as
        # fully_open with low openness) doesn't false-fire.
        #
        # All four conditions must hold:
        #   - thumb.state == "fully_open" (classifier's call)
        #   - thumb.openness >= 0.65          (visibly open, not borderline)
        #   - thumb.curl    <= 0.35           (not curled across the palm)
        #   - spreads.thumb_index distance >= 0.55
        #     (the thumb is actually splayed away from the index;
        #      a folded thumb sits close to the index regardless
        #      of what the classifier thinks)
        #
        # User screenshot showed the failing case: state=fully_open,
        # openness=0.54, curl=0.46, spread=0.46 â€” all four signals
        # in the borderline band. The new AND-of-four rejects it.
        thumb = hand_reading.fingers.get("thumb")
        thumb_state = getattr(thumb, "state", None) if thumb is not None else None
        thumb_openness = float(getattr(thumb, "openness", 0.0) or 0.0) if thumb is not None else 0.0
        thumb_curl = float(getattr(thumb, "curl", 0.0) or 0.0) if thumb is not None else 1.0
        spread_ti = hand_reading.spreads.get("thumb_index") if hasattr(hand_reading, "spreads") else None
        thumb_index_distance = float(getattr(spread_ti, "distance", 0.0) or 0.0) if spread_ti is not None else 0.0
        thumb_open_now = (
            thumb is not None
            and thumb_state == "fully_open"
            and thumb_openness >= 0.65
            and thumb_curl <= 0.35
            and thumb_index_distance >= 0.55
        )

        erase_active = self._drawing_erase_pose_active(prediction, hand_reading)
        draw_active = self._drawing_draw_pose_active(prediction, hand_reading)
        if thumb_open_now:
            # Decisive thumb-open invalidates draw / erase pose
            # immediately, regardless of finger geometry. The user
            # has clearly indicated they're not gripping a pen.
            draw_active = False
            erase_active = False
        # Tell the stroke renderer to freeze the pen tip wherever it
        # was on the LAST drawing-pose frame, so the natural hand
        # motion that accompanies opening the thumb doesn't drift the
        # stroke endpoint to the right before the 0.20 s commit fires.
        self._drawing_freeze_pen = bool(thumb_open_now)

        # Track sustained draw-pose intent. Single-frame draw_active
        # blips (e.g. dropping an open hand briefly reads as draw
        # pose for one frame as the wrist swings down) increment
        # the counter to 1 only â€” and we require 2 to actually
        # start a new stroke. Once the user is mid-stroke (grace
        # window is active) we extend per-frame as before so tilt
        # / rotation wobble doesn't break the stroke.
        if draw_active:
            self._drawing_draw_active_streak += 1
        else:
            self._drawing_draw_active_streak = 0

        if erase_active:
            # Erase grace: 0.40 s (was 0.25). The longer window
            # gives more tolerance for tilt and brief misreads
            # without losing the erase pose.
            self._drawing_erase_grace_until = now + 0.40
        already_drawing = now < self._drawing_draw_grace_until
        if draw_active and (already_drawing or self._drawing_draw_active_streak >= 2):
            # Either continuing an active stroke (already in grace)
            # or sustained draw pose for >=2 frames (new stroke) â€”
            # both extend grace by 0.40 s. Anti-misfire: a single
            # frame draw_active blip never extends grace, so a
            # passing hand-drop can't start an unintended stroke.
            self._drawing_draw_grace_until = now + 0.40

        # Time-based pen-lift hold. First open frame stamps
        # _drawing_thumb_open_since; first not-open frame resets
        # it. When elapsed >= hold, commit any in-flight stroke
        # and cancel both grace windows.
        if thumb_open_now:
            if self._drawing_thumb_open_since <= 0.0:
                self._drawing_thumb_open_since = now
        else:
            self._drawing_thumb_open_since = 0.0
        if (
            self._drawing_thumb_open_since > 0.0
            and (now - self._drawing_thumb_open_since)
            >= self._DRAWING_THUMB_OPEN_HOLD_SECONDS
        ):
            if (
                self._drawing_render_target == "camera"
                and self._camera_draw_active_stroke_points
            ):
                self._commit_camera_draw_stroke()
            self._drawing_draw_grace_until = 0.0
            self._drawing_erase_grace_until = 0.0
            self._drawing_draw_active_streak = 0
            # Reset the timer so a single hold counts as one lift
            # event, not a continuously-firing one. The user has to
            # close the thumb (or its openness drops below 0.55)
            # and re-open to trigger another lift.
            self._drawing_thumb_open_since = 0.0

        if erase_active or now < self._drawing_erase_grace_until:
            self._drawing_tool = "erase"
            self._drawing_control_text = f"drawing erase ({self._drawing_render_target})"
            self._camera_draw_last_point = None
            return
        if draw_active or now < self._drawing_draw_grace_until:
            self._drawing_tool = "draw"
            self._drawing_control_text = f"drawing ({self._drawing_render_target})"
            return
        if self._drawing_pinch_pose_active(hand_reading):
            self._drawing_tool = "grab"
            self._drawing_control_text = f"drawing grab ({self._drawing_render_target})"
            self._camera_draw_last_point = None
            return
        self._drawing_tool = "hover"

        self._drawing_control_text = f"drawing hover ({self._drawing_render_target})"
        self._camera_draw_last_point = None

    def _reset_drawing_wheel(self, *, clear_cooldown: bool = False) -> None:
        self._drawing_wheel_visible = False
        self._drawing_wheel_anchor = None
        self._drawing_wheel_cursor_offset = None
        self._drawing_wheel_selected_key = None
        self._drawing_wheel_selected_since = 0.0
        self._drawing_wheel_pose_grace_until = 0.0
        self._drawing_wheel_candidate = "neutral"
        self._drawing_wheel_candidate_since = 0.0
        if clear_cooldown:
            self._drawing_wheel_cooldown_until = 0.0
        if self.drawing_wheel_overlay.isVisible():
            self.drawing_wheel_overlay.hide_overlay()

    def _drawing_wheel_items(self) -> tuple[tuple[str, str, float], ...]:
        return (
            ("switch_view", "Switch View", 90.0),
            ("save", "Save Drawing", 18.0),
            ("shape", "Shape Mode", 306.0),
            ("pen_options", "Pen Options", 234.0),
            ("eraser_options", "Eraser Options", 162.0),
        )

    def _drawing_wheel_label(self, key: str) -> str:
        for item_key, label, _angle in self._drawing_wheel_items():
            if item_key == key:
                return label
        return key.replace("_", " ")

    def _queue_drawing_request(self, action: str) -> None:
        self._drawing_request_token += 1
        self._drawing_request_action = str(action or "")

    def _execute_drawing_wheel_action(self, key: str) -> None:
        if key == "switch_view":
            if self._drawing_render_target == "camera":
                self._commit_camera_draw_stroke()
                self._camera_draw_last_point = None
                self._camera_draw_erasing = False
            self._drawing_render_target = "camera" if self._drawing_render_target == "screen" else "screen"
            self._drawing_control_text = f"drawing target {self._drawing_render_target}"
            self.command_detected.emit(self._drawing_control_text)
            self._record_action(f"drawing_target_{self._drawing_render_target}", self._drawing_control_text)
            return
        if key == "pen_options":
            self._queue_drawing_request("pen_options")
            self._drawing_control_text = "drawing pen options"
            self.command_detected.emit("Drawing pen options")
            self._record_action("drawing_pen_options", "drawing pen options opened")
            return
        if key == "eraser_options":
            self._queue_drawing_request("eraser_options")
            self._drawing_control_text = "drawing eraser options"
            self.command_detected.emit("Drawing eraser options")
            self._record_action("drawing_eraser_options", "drawing eraser options opened")
            return
        if key == "save":
            self._queue_drawing_request("save")
            self._drawing_control_text = "drawing save"
            self.command_detected.emit("Saving drawing")
            self._record_action("drawing_save", "drawing saved")
            return
        if key == "shape":
            self._drawing_shape_mode = not self._drawing_shape_mode
            self._queue_drawing_request("shape_on" if self._drawing_shape_mode else "shape_off")
            self._drawing_control_text = "Shape mode on" if self._drawing_shape_mode else "Shape mode off"
            hint_text = "Shape mode: ON" if self._drawing_shape_mode else "Shape mode: OFF"
            try:
                self.voice_status_overlay.show_info_hint(hint_text, duration=3.0)
            except Exception:
                pass
            self.command_detected.emit(self._drawing_control_text)
            self._record_action(
                "drawing_shape_on" if self._drawing_shape_mode else "drawing_shape_off",
                self._drawing_control_text,
            )
            return
        self._drawing_control_text = "drawing wheel action"
        self.command_detected.emit(self._drawing_control_text)

    def _update_drawing_wheel_selection(self, hand_reading, now: float) -> None:
        if self._drawing_wheel_anchor is None:
            self._drawing_wheel_anchor = hand_reading.palm.center.copy()
        offset = (hand_reading.palm.center - self._drawing_wheel_anchor) / max(hand_reading.palm.scale, 1e-6)
        self._drawing_wheel_cursor_offset = (float(offset[0]), float(offset[1]))
        selection_key = self._wheel_selection_key(float(offset[0]), float(offset[1]), self._drawing_wheel_items())
        if selection_key is None:
            self._drawing_wheel_selected_key = None
            self._drawing_wheel_selected_since = now
            self._drawing_control_text = "drawing wheel active"
            return
        if selection_key != self._drawing_wheel_selected_key:
            self._drawing_wheel_selected_key = selection_key
            self._drawing_wheel_selected_since = now
            self._drawing_control_text = f"drawing wheel: {self._drawing_wheel_label(selection_key)}"
            return
        if now - self._drawing_wheel_selected_since < 0.85:
            return
        self._execute_drawing_wheel_action(selection_key)
        self._drawing_wheel_cooldown_until = now + 1.5
        self._reset_drawing_wheel()

    def _update_drawing_wheel(self, prediction, hand_reading, now: float, *, active: bool) -> bool:
        if not active or prediction is None or hand_reading is None:
            if self._drawing_wheel_visible and now >= self._drawing_wheel_pose_grace_until:
                self._drawing_control_text = "drawing wheel closed"
                self._reset_drawing_wheel()
            else:
                self._drawing_wheel_candidate = "neutral"
                self._drawing_wheel_candidate_since = now
            return self._drawing_wheel_visible
        wheel_pose = self._drawing_wheel_pose_active(prediction)
        if self._drawing_wheel_visible:
            if wheel_pose:
                self._drawing_wheel_pose_grace_until = now + 0.25
                self._update_drawing_wheel_selection(hand_reading, now)
            elif now >= self._drawing_wheel_pose_grace_until:
                self._drawing_control_text = "drawing wheel closed"
                self._reset_drawing_wheel()
            return True
        if now < self._drawing_wheel_cooldown_until:
            if not wheel_pose:
                self._drawing_wheel_candidate = "neutral"
            return False
        if not wheel_pose:
            self._drawing_wheel_candidate = "neutral"
            self._drawing_wheel_candidate_since = now
            return False
        if self._drawing_wheel_candidate != "wheel_pose":
            self._drawing_wheel_candidate = "wheel_pose"
            self._drawing_wheel_candidate_since = now
            self._drawing_control_text = "hold wheel pose for drawing settings"
            return True
        if now - self._drawing_wheel_candidate_since < 0.85:
            return True
        self._drawing_wheel_visible = True
        self._drawing_wheel_anchor = hand_reading.palm.center.copy()
        self._drawing_wheel_cursor_offset = (0.0, 0.0)
        self._drawing_wheel_selected_key = None
        self._drawing_wheel_selected_since = now
        self._drawing_wheel_pose_grace_until = now + 0.25
        self._drawing_control_text = "drawing wheel active"
        return True

    def get_camera_draw_canvas_snapshot(self):
        """Return a numpy copy of the current camera-target drawing
        canvas (BGRA) without doing any compositing or disk I/O.
        Callable from the UI thread; cheap (just np.copy of a
        camera-sized array). Returns None when there's no canvas."""
        canvas = self._camera_draw_canvas
        if canvas is None:
            return None
        if self._camera_draw_active_stroke_points:
            try:
                self._commit_camera_draw_stroke()
            except Exception:
                pass
        try:
            return canvas.copy()
        except Exception:
            return None

    def save_camera_draw_snapshot(self, target_path) -> bool:
        """Write the current camera-target drawing canvas to a PNG.

        Called from the UI thread when the save wheel action fires while the
        drawing render target is ``camera``. The screen overlay is empty in
        that case, so without this the save would produce a blank PNG.
        """
        canvas = self._camera_draw_canvas
        if canvas is None:
            return False
        if self._camera_draw_active_stroke_points:
            self._commit_camera_draw_stroke()
        try:
            snapshot = canvas.copy()
        except Exception:
            return False
        try:
            height, width = snapshot.shape[:2]
            alpha = snapshot[:, :, 3:4].astype(np.float32) / 255.0
            fg = snapshot[:, :, :3].astype(np.float32)
            # Pick a background that contrasts with the average stroke color â€”
            # mirrors the screen-overlay save so dark pens on dark bg stay readable.
            mean_stroke_bgr = fg.sum(axis=(0, 1)) / max(float(alpha.sum()) * 3.0, 1.0)
            bg_color = (255.0, 255.0, 255.0) if float(mean_stroke_bgr.mean()) < 60.0 else (0.0, 0.0, 0.0)
            bg = np.zeros((height, width, 3), dtype=np.float32)
            bg[:, :, 0] = bg_color[0]
            bg[:, :, 1] = bg_color[1]
            bg[:, :, 2] = bg_color[2]
            composite = (fg * alpha + bg * (1.0 - alpha)).clip(0.0, 255.0).astype(np.uint8)
            from pathlib import Path as _Path
            path = _Path(str(target_path))
            path.parent.mkdir(parents=True, exist_ok=True)
            return bool(cv2.imwrite(str(path), composite))
        except Exception:
            return False

    def acknowledge_drawing_request(self, token: int | None = None) -> None:
        try:
            current_token = int(self._drawing_request_token)
        except Exception:
            current_token = 0
        if token is not None:
            try:
                requested_token = int(token)
            except Exception:
                requested_token = -1
            if requested_token != current_token:
                return
        self._drawing_request_action = ""

    def _reset_utility_wheel(self, *, clear_cooldown: bool = False) -> None:
        self._utility_wheel_candidate = "neutral"
        self._utility_wheel_candidate_since = 0.0
        self._utility_wheel_visible = False
        self._utility_wheel_anchor = None
        self._utility_wheel_selected_key = None
        self._utility_wheel_selected_since = 0.0
        self._utility_wheel_pose_grace_until = 0.0
        self._utility_wheel_cursor_offset = None
        if clear_cooldown:
            self._utility_wheel_cooldown_until = 0.0
        if self.utility_wheel_overlay.isVisible():
            self.utility_wheel_overlay.hide_overlay()

    def _utility_wheel_items(self) -> tuple[tuple[str, str, float], ...]:
        labels = (
            ("screenshot", "Screenshot"),
            ("screenshot_custom", "Screenshot Custom"),
            ("screen_record", "Screen Record"),
            ("screen_record_custom", "Screen Record Custom"),
            ("clip_30s", "Clip 30 Sec"),
            ("clip_1m", "Clip 1 Min"),
        )
        slice_span = 360.0 / len(labels)
        return tuple(
            (key, label, (90.0 - index * slice_span) % 360.0)
            for index, (key, label) in enumerate(labels)
        )

    def _utility_wheel_label(self, key: str) -> str:
        for item_key, label, _angle in self._utility_wheel_items():
            if item_key == key:
                return label
        return key.replace("_", " ")

    def _utility_wheel_pose_active(self, hand_reading) -> bool:
        if hand_reading is None:
            return False
        fingers = hand_reading.fingers
        return (
            self._drawing_finger_extendedish(fingers.get("index"), primary=True)
            and self._drawing_finger_extendedish(fingers.get("pinky"))
            and self._drawing_finger_foldedish(fingers.get("thumb"))
            and self._drawing_finger_foldedish(fingers.get("middle"))
            and self._drawing_finger_foldedish(fingers.get("ring"))
        )

    def _queue_utility_request(self, action: str) -> None:
        self._utility_request_token += 1
        self._utility_request_action = str(action or "")

    def acknowledge_utility_request(self, token: int | None = None) -> None:
        try:
            current_token = int(self._utility_request_token)
        except Exception:
            current_token = 0
        if token is not None:
            try:
                requested_token = int(token)
            except Exception:
                requested_token = -1
            if requested_token != current_token:
                return
        self._utility_request_action = ""

    def _execute_utility_wheel_action(self, key: str) -> None:
        if key == "screenshot":
            self._queue_utility_request("screenshot_full")
            self.command_detected.emit("Full screenshot in 3 seconds")
            return
        if key == "screenshot_custom":
            self._queue_utility_request("screenshot_custom")
            self.command_detected.emit("Drag to choose screenshot area")
            return
        if key == "screen_record":
            self._queue_utility_request("record_full")
            self.command_detected.emit("Full screen record in 3 seconds")
            return
        if key == "screen_record_custom":
            self._queue_utility_request("record_custom")
            self.command_detected.emit("Drag to choose recording area")
            return
        if key == "clip_30s":
            self._queue_utility_request("clip_30s")
            self.command_detected.emit("Saving last 30 seconds")
            return
        if key == "clip_1m":
            self._queue_utility_request("clip_1m")
            self.command_detected.emit("Saving last 1 minute")
            return
        self.command_detected.emit(f"{self._utility_wheel_label(key)} coming soon")

    def _update_utility_wheel_selection(self, hand_reading, now: float) -> None:
        if self._utility_wheel_anchor is None:
            self._utility_wheel_anchor = hand_reading.palm.center.copy()
        offset = (hand_reading.palm.center - self._utility_wheel_anchor) / max(hand_reading.palm.scale, 1e-6)
        self._utility_wheel_cursor_offset = (float(offset[0]), float(offset[1]))
        selection_key = self._wheel_selection_key(float(offset[0]), float(offset[1]), self._utility_wheel_items())
        if selection_key is None:
            self._utility_wheel_selected_key = None
            self._utility_wheel_selected_since = now
            return
        if selection_key != self._utility_wheel_selected_key:
            self._utility_wheel_selected_key = selection_key
            self._utility_wheel_selected_since = now
            return
        if now - self._utility_wheel_selected_since < 0.85:
            return
        self._execute_utility_wheel_action(selection_key)
        self._utility_wheel_cooldown_until = now + 1.5
        self._reset_utility_wheel()

    def _update_utility_wheel(self, hand_reading, hand_handedness: str | None, now: float) -> bool:
        active = hand_handedness == "Right" and hand_reading is not None
        if not active:
            if self._utility_wheel_visible and now >= self._utility_wheel_pose_grace_until:
                self._reset_utility_wheel()
            else:
                self._utility_wheel_candidate = "neutral"
                self._utility_wheel_candidate_since = now
            return self._utility_wheel_visible
        wheel_pose = self._utility_wheel_pose_active(hand_reading)
        if self._utility_wheel_visible:
            if wheel_pose:
                self._utility_wheel_pose_grace_until = now + 0.25
                self._update_utility_wheel_selection(hand_reading, now)
            elif now >= self._utility_wheel_pose_grace_until:
                self._reset_utility_wheel()
            return True
        if now < self._utility_wheel_cooldown_until:
            if not wheel_pose:
                self._utility_wheel_candidate = "neutral"
            return False
        if not wheel_pose:
            self._utility_wheel_candidate = "neutral"
            self._utility_wheel_candidate_since = now
            return False
        if self._utility_wheel_candidate != "utility_wheel_pose":
            self._utility_wheel_candidate = "utility_wheel_pose"
            self._utility_wheel_candidate_since = now
            return True
        if now - self._utility_wheel_candidate_since < 0.85:
            return True
        self._utility_wheel_visible = True
        self._utility_wheel_anchor = hand_reading.palm.center.copy()
        self._utility_wheel_cursor_offset = (0.0, 0.0)
        self._utility_wheel_selected_key = None
        self._utility_wheel_selected_since = now
        self._utility_wheel_pose_grace_until = now + 0.25
        return True

    def set_utility_recording_active(self, active: bool) -> None:
        self._utility_recording_active = bool(active)
        if not self._utility_recording_active:
            self._utility_recording_stop_candidate_since = 0.0

    def set_utility_capture_selection_active(self, active: bool) -> None:
        self._utility_capture_selection_active = bool(active)
        self._utility_capture_clicks_armed = False
        if not self._utility_capture_selection_active:
            self._utility_capture_cursor_norm = None
            self._utility_capture_left_down = False
            self._utility_capture_right_down = False
            self._utility_capture_box_bounds = None

    def _utility_capture_click_down(self, finger) -> bool:
        if finger is None:
            return False
        openness = float(getattr(finger, 'openness', 0.0) or 0.0)
        curl = float(getattr(finger, 'curl', 0.0) or 0.0)
        return finger.state in {'closed', 'mostly_curled'} or openness <= 0.42 or curl >= 0.52

    # Sensitivity boost applied while a hand-selector dialog (pen
    # options, eraser options, monitor selection, etc.) is open.
    # The palm is mapped through the same ergonomic mouse-control
    # box as regular mouse mode, but the effective box is shrunk
    # around its center by this factor so the user can reach the
    # full dialog with less hand travel. 1.0 = same as mouse mode;
    # <1.0 = more sensitive (smaller hand reach covers same screen
    # area).
    _UTILITY_CAPTURE_SENSITIVITY = 0.62

    def _update_utility_capture_selection(self, hand_reading, hand_handedness: str | None) -> None:
        if hand_handedness != 'Right' or hand_reading is None:
            self._utility_capture_cursor_norm = None
            self._utility_capture_left_down = False
            self._utility_capture_right_down = False
            self._utility_capture_clicks_armed = False
            self._utility_capture_box_bounds = None
            return
        try:
            palm_center = getattr(hand_reading.palm, 'center', None)
            if palm_center is None or len(palm_center) < 2:
                raise ValueError('missing palm center')
            palm_x = float(palm_center[0])
            palm_y = float(palm_center[1])
            # Route the palm through the regular mouse-control box
            # (so the dialog cursor uses the same ergonomic hand-reach
            # region the user already learned for mouse mode), then
            # shrink that box around its center by the sensitivity
            # multiplier. Smaller effective box = same hand motion
            # reaches further on screen = "more sensitive".
            try:
                min_x, min_y, max_x, max_y = self.mouse_tracker._camera_bounds()
            except Exception:
                min_x, min_y, max_x, max_y = (0.0, 0.0, 1.0, 1.0)
            cx = (min_x + max_x) * 0.5
            cy = (min_y + max_y) * 0.5
            half_w = max(1e-3, (max_x - min_x) * 0.5 * self._UTILITY_CAPTURE_SENSITIVITY)
            half_h = max(1e-3, (max_y - min_y) * 0.5 * self._UTILITY_CAPTURE_SENSITIVITY)
            eff_min_x = max(0.0, cx - half_w)
            eff_max_x = min(1.0, cx + half_w)
            eff_min_y = max(0.0, cy - half_h)
            eff_max_y = min(1.0, cy + half_h)
            cursor_x = max(0.0, min(1.0, (palm_x - eff_min_x) / max(1e-6, eff_max_x - eff_min_x)))
            cursor_y = max(0.0, min(1.0, (palm_y - eff_min_y) / max(1e-6, eff_max_y - eff_min_y)))
            # Stash the shrunk bounds so the live-view widget can
            # render the same red box during chooser dialogs as it
            # does in mouse mode — the user asked for it to "still
            # use the red mouse control area" while these dialogs
            # are open.
            self._utility_capture_box_bounds = (eff_min_x, eff_min_y, eff_max_x, eff_max_y)
        except Exception:
            self._utility_capture_cursor_norm = None
            self._utility_capture_left_down = False
            self._utility_capture_right_down = False
            self._utility_capture_clicks_armed = False
            self._utility_capture_box_bounds = None
            return
        fingers = hand_reading.fingers
        self._utility_capture_cursor_norm = (cursor_x, cursor_y)
        index_down = self._utility_capture_click_down(fingers.get('index'))
        middle_down = self._utility_capture_click_down(fingers.get('middle'))
        if not self._utility_capture_clicks_armed:
            self._utility_capture_left_down = False
            self._utility_capture_right_down = False
            if not index_down and not middle_down:
                self._utility_capture_clicks_armed = True
            return
        self._utility_capture_left_down = index_down and not middle_down
        self._utility_capture_right_down = middle_down and not index_down

    def set_drawing_brush(self, color: str | None = None, thickness: int | None = None) -> None:
        old_color = self._drawing_brush_hex
        old_thickness = self._drawing_brush_thickness
        if color:
            self._drawing_brush_hex = str(color)
        if thickness is not None:
            self._drawing_brush_thickness = int(max(2, thickness))
        # Log pen changes to recent-actions so the user sees the
        # exact moment color / thickness flipped while drawing.
        if color and color != old_color:
            self._record_action("drawing_pen_color", f"pen color {self._drawing_brush_hex}")
        if thickness is not None and thickness != old_thickness:
            self._record_action(
                "drawing_pen_thickness", f"pen thickness {self._drawing_brush_thickness}"
            )

    def set_drawing_eraser(self, thickness: int | None = None, mode: str | None = None) -> None:
        old_thickness = self._drawing_eraser_thickness
        old_mode = self._drawing_eraser_mode
        if thickness is not None:
            self._drawing_eraser_thickness = int(max(4, thickness))
        if mode:
            normalized = str(mode).strip().lower()
            if normalized in {"normal", "stroke"}:
                self._drawing_eraser_mode = normalized
        if thickness is not None and thickness != old_thickness:
            self._record_action(
                "drawing_eraser_thickness", f"eraser thickness {self._drawing_eraser_thickness}"
            )
        if mode and mode != old_mode:
            self._record_action(
                "drawing_eraser_mode", f"eraser mode {self._drawing_eraser_mode}"
            )

    def _drawing_brush_bgr(self) -> tuple[int, int, int]:
        value = str(self._drawing_brush_hex or "#FFFFFF").strip().lstrip("#")
        if len(value) != 6:
            return (255, 255, 255)
        try:
            r = int(value[0:2], 16); g = int(value[2:4], 16); b = int(value[4:6], 16)
        except Exception:
            return (255, 255, 255)
        return (b, g, r)

    def _ensure_camera_draw_canvas(self, frame_shape) -> None:
        height, width = frame_shape[:2]
        if height <= 0 or width <= 0:
            return
        if self._camera_draw_canvas is not None and self._camera_draw_canvas.shape[:2] == (height, width):
            return
        self._camera_draw_canvas = np.zeros((height, width, 4), dtype=np.uint8)
        self._camera_draw_history = []
        self._camera_draw_strokes = []
        self._camera_draw_active_stroke_points = []
        self._camera_draw_raster_dirty = False
        self._camera_draw_last_point = None
        self._camera_draw_erasing = False

    def _clone_camera_draw_strokes(self) -> list[dict]:
        clones: list[dict] = []
        for stroke in self._camera_draw_strokes:
            clones.append(
                {
                    "color": tuple(int(v) for v in stroke.get("color", (255, 255, 255))),
                    "thickness": int(stroke.get("thickness", 2)),
                    "points": [(float(x), float(y)) for x, y in stroke.get("points", [])],
                }
            )
        return clones

    @staticmethod
    def _bezier_segment_steps(p0: tuple[int, int], p2: tuple[int, int]) -> int:
        # Sample density follows the chord length so short segments
        # cost ~8 samples and long segments cap at ~24. Tighter than
        # that wastes cycles; looser leaves visible facets at high
        # thickness.
        dx = p2[0] - p0[0]
        dy = p2[1] - p0[1]
        rough = (dx * dx + dy * dy) ** 0.5
        return max(8, min(24, int(rough / 4.0) + 8))

    @classmethod
    def _quad_bezier_polyline(
        cls,
        p_start: tuple[int, int],
        p_ctrl: tuple[int, int],
        p_end: tuple[int, int],
    ) -> "np.ndarray":
        steps = cls._bezier_segment_steps(p_start, p_end)
        pts = np.empty((steps + 1, 2), dtype=np.int32)
        for i in range(steps + 1):
            t = i / steps
            u = 1.0 - t
            x = u * u * p_start[0] + 2.0 * u * t * p_ctrl[0] + t * t * p_end[0]
            y = u * u * p_start[1] + 2.0 * u * t * p_ctrl[1] + t * t * p_end[1]
            pts[i, 0] = int(round(x))
            pts[i, 1] = int(round(y))
        return pts

    @classmethod
    def _draw_smooth_polyline_canvas(
        cls,
        canvas: "np.ndarray",
        points: list[tuple[float, float]],
        color_bgra: tuple[int, int, int, int],
        thickness: int,
    ) -> None:
        # Quadratic Bezier through midpoints (Procreate-style smoothing):
        # each raw sample becomes a control point, the curve passes
        # through the midpoints between consecutive samples. Renders
        # the same path that incremental live drawing builds, so
        # canvas state stays consistent across undo/erase rerenders.
        n = len(points)
        if n < 2:
            return
        rounded = [
            (int(round(float(x))), int(round(float(y)))) for x, y in points
        ]
        if n == 2:
            cv2.line(canvas, rounded[0], rounded[1], color_bgra, thickness, cv2.LINE_AA)
            return
        mid01 = (
            (rounded[0][0] + rounded[1][0]) // 2,
            (rounded[0][1] + rounded[1][1]) // 2,
        )
        cv2.line(canvas, rounded[0], mid01, color_bgra, thickness, cv2.LINE_AA)
        for i in range(1, n - 1):
            pa = rounded[i - 1]
            pb = rounded[i]
            pc = rounded[i + 1]
            start = ((pa[0] + pb[0]) // 2, (pa[1] + pb[1]) // 2)
            end = ((pb[0] + pc[0]) // 2, (pb[1] + pc[1]) // 2)
            arr = cls._quad_bezier_polyline(start, pb, end)
            cv2.polylines(canvas, [arr], False, color_bgra, thickness, cv2.LINE_AA)
        last = rounded[-1]
        prev = rounded[-2]
        mid_last = ((prev[0] + last[0]) // 2, (prev[1] + last[1]) // 2)
        cv2.line(canvas, mid_last, last, color_bgra, thickness, cv2.LINE_AA)

    def _camera_draw_rerender_from_strokes(self) -> None:
        if self._camera_draw_canvas is None:
            return
        self._camera_draw_canvas[:, :, :] = 0
        for stroke in self._camera_draw_strokes:
            points = stroke.get("points") or []
            if len(points) < 2:
                continue
            color = tuple(int(v) for v in stroke.get("color", (255, 255, 255)))
            thickness = int(max(1, stroke.get("thickness", 2)))
            self._draw_smooth_polyline_canvas(
                self._camera_draw_canvas,
                points,
                (*color, 255),
                thickness,
            )

    @staticmethod
    def _camera_point_to_segment_distance_sq(px: float, py: float, ax: float, ay: float, bx: float, by: float) -> float:
        abx = bx - ax
        aby = by - ay
        if abs(abx) < 1e-9 and abs(aby) < 1e-9:
            dx = px - ax
            dy = py - ay
            return dx * dx + dy * dy
        apx = px - ax
        apy = py - ay
        denom = abx * abx + aby * aby
        t = max(0.0, min(1.0, (apx * abx + apy * aby) / denom))
        cx = ax + t * abx
        cy = ay + t * aby
        dx = px - cx
        dy = py - cy
        return dx * dx + dy * dy

    def _camera_draw_find_stroke_at(self, px: float, py: float, radius: float) -> int | None:
        for idx in range(len(self._camera_draw_strokes) - 1, -1, -1):
            if self._camera_draw_stroke_hits_position(self._camera_draw_strokes[idx], px, py, radius):
                return idx
        return None

    def _handle_drawing_grab(self, point: tuple[int, int]) -> None:
        px = float(point[0])
        py = float(point[1])
        if self._drawing_grabbed_stroke_index is None:
            radius = max(12.0, float(self._drawing_brush_thickness) * 1.2)
            hit = self._camera_draw_find_stroke_at(px, py, radius)
            if hit is None:
                self._drawing_grab_last_point = point
                return
            if self._camera_draw_raster_dirty and self._camera_draw_strokes:
                self._camera_draw_raster_dirty = False
                self._camera_draw_rerender_from_strokes()
            if not self._drawing_grab_history_pushed:
                self._camera_draw_push_history()
                self._drawing_grab_history_pushed = True
            self._drawing_grabbed_stroke_index = hit
            self._drawing_grab_last_point = point
            return
        last = self._drawing_grab_last_point
        if last is None:
            self._drawing_grab_last_point = point
            return
        dx = point[0] - last[0]
        dy = point[1] - last[1]
        self._drawing_grab_last_point = point
        if dx == 0 and dy == 0:
            return
        idx = self._drawing_grabbed_stroke_index
        if idx is None or idx < 0 or idx >= len(self._camera_draw_strokes):
            self._release_drawing_grab()
            return
        stroke = self._camera_draw_strokes[idx]
        points = stroke.get("points") or []
        stroke["points"] = [(float(x) + dx, float(y) + dy) for x, y in points]
        self._camera_draw_rerender_from_strokes()

    def _camera_draw_stroke_hits_position(self, stroke: dict, px: float, py: float, radius: float) -> bool:
        points = stroke.get("points") or []
        if not points:
            return False
        threshold = max(float(radius), float(stroke.get("thickness", 0)) * 0.5 + 2.0)
        limit_sq = threshold * threshold
        if len(points) == 1:
            sx, sy = points[0]
            dx = sx - px
            dy = sy - py
            return dx * dx + dy * dy <= limit_sq
        for (ax, ay), (bx, by) in zip(points, points[1:]):
            if self._camera_point_to_segment_distance_sq(px, py, float(ax), float(ay), float(bx), float(by)) <= limit_sq:
                return True
        return False

    def _commit_camera_draw_stroke(self) -> None:
        if not self._camera_draw_active_stroke_points:
            return
        points = list(self._camera_draw_active_stroke_points)
        brush = tuple(int(v) for v in self._drawing_brush_bgr())
        thickness = int(max(2, self._drawing_brush_thickness))
        # Live drawing left the canvas ending at mid(P_{N-1}, P_N).
        # Add the trailing stub mid -> P_N here so the canvas matches
        # what _camera_draw_rerender_from_strokes would produce â€”
        # otherwise an erase / undo pass would visibly nudge the
        # stroke endpoint.
        if (
            self._camera_draw_canvas is not None
            and len(points) >= 2
        ):
            (px1, py1), (px2, py2) = points[-2], points[-1]
            mid = (
                int(round((px1 + px2) / 2.0)),
                int(round((py1 + py2) / 2.0)),
            )
            end = (int(round(px2)), int(round(py2)))
            cv2.line(
                self._camera_draw_canvas,
                mid,
                end,
                (*brush, 255),
                thickness,
                cv2.LINE_AA,
            )
        if len(points) == 1:
            x, y = points[0]
            points.append((x + 0.01, y + 0.01))
        self._camera_draw_strokes.append(
            {
                "color": brush,
                "thickness": thickness,
                "points": points,
            }
        )
        self._camera_draw_active_stroke_points = []

    def _camera_draw_point(self, frame_shape) -> tuple[int, int] | None:
        if self._drawing_cursor_norm is None:
            return None
        height, width = frame_shape[:2]
        x = int(round(max(0.0, min(1.0, float(self._drawing_cursor_norm[0]))) * max(width - 1, 1)))
        y = int(round(max(0.0, min(1.0, float(self._drawing_cursor_norm[1]))) * max(height - 1, 1)))
        return x, y

    def _camera_draw_push_history(self) -> None:
        if self._camera_draw_canvas is None:
            return
        self._camera_draw_history.append(
            (
                self._camera_draw_canvas.copy(),
                self._clone_camera_draw_strokes(),
                bool(self._camera_draw_raster_dirty),
            )
        )
        if len(self._camera_draw_history) > 24:
            self._camera_draw_history = self._camera_draw_history[-24:]

    def _camera_draw_undo(self) -> bool:
        if not self._camera_draw_history:
            return False
        canvas, strokes, raster_dirty = self._camera_draw_history.pop()
        self._camera_draw_canvas = canvas
        self._camera_draw_strokes = strokes
        self._camera_draw_raster_dirty = bool(raster_dirty)
        self._camera_draw_active_stroke_points = []
        self._camera_draw_last_point = None
        self._camera_draw_erasing = False
        return True

    def _perform_drawing_swipe_action(self, direction: str) -> bool:
        key = str(direction or "").strip().lower()
        if key not in {"swipe_left", "swipe_right"}:
            return False
        if self._drawing_render_target == "camera":
            self._ensure_camera_draw_canvas((480, 640, 3))
            if key == "swipe_left":
                success = self._camera_draw_undo()
                self._drawing_control_text = "drawing undo" if success else "drawing nothing to undo"
            else:
                if self._camera_draw_canvas is not None:
                    self._camera_draw_push_history()
                    self._camera_draw_canvas[:, :, :] = 0
                    self._camera_draw_strokes = []
                    self._camera_draw_active_stroke_points = []
                    self._camera_draw_raster_dirty = False
                    self._camera_draw_last_point = None
                success = True
                self._drawing_control_text = "drawing cleared"
            self.command_detected.emit(self._drawing_control_text)
            self._record_action(
                "drawing_undo" if key == "swipe_left" else "drawing_clear",
                self._drawing_control_text,
            )
            return success
        self._queue_drawing_request("undo" if key == "swipe_left" else "clear")
        self._drawing_control_text = "drawing undo" if key == "swipe_left" else "drawing cleared"
        self.command_detected.emit(self._drawing_control_text)
        self._record_action(
            "drawing_undo" if key == "swipe_left" else "drawing_clear",
            self._drawing_control_text,
        )
        return True

    def _release_drawing_grab(self) -> None:
        self._drawing_grabbed_stroke_index = None
        self._drawing_grab_last_point = None
        self._drawing_grab_history_pushed = False
        self._release_drawing_stretch()

    def _release_drawing_stretch(self) -> None:
        self._drawing_stretch_active = False
        self._drawing_stretch_initial_distance = None
        self._drawing_stretch_initial_points = None
        self._drawing_stretch_centroid = None
        self._drawing_stretch_history_pushed = False

    def _drawing_secondary_pinch_point(self, frame_shape) -> tuple[int, int] | None:
        secondary = self._drawing_secondary_hand_reading
        if secondary is None:
            return None
        if not self._drawing_pinch_pose_active(secondary):
            return None
        try:
            cx = max(0.0, min(1.0, float(secondary.landmarks[8][0])))
            cy = max(0.0, min(1.0, float(secondary.landmarks[8][1])))
        except Exception:
            return None
        height, width = frame_shape[:2]
        x = int(round(cx * max(width - 1, 1)))
        y = int(round(cy * max(height - 1, 1)))
        return x, y

    def _handle_drawing_stretch(self, primary_point: tuple[int, int], secondary_point: tuple[int, int]) -> None:
        idx = self._drawing_grabbed_stroke_index
        if idx is None or idx < 0 or idx >= len(self._camera_draw_strokes):
            self._release_drawing_grab()
            return
        stroke = self._camera_draw_strokes[idx]
        dx = float(secondary_point[0] - primary_point[0])
        dy = float(secondary_point[1] - primary_point[1])
        distance = (dx * dx + dy * dy) ** 0.5
        if distance < 6.0:
            return
        if not self._drawing_stretch_active:
            points = [(float(x), float(y)) for x, y in (stroke.get("points") or [])]
            if not points:
                return
            if self._camera_draw_raster_dirty and self._camera_draw_strokes:
                self._camera_draw_raster_dirty = False
                self._camera_draw_rerender_from_strokes()
            if not self._drawing_stretch_history_pushed and not self._drawing_grab_history_pushed:
                self._camera_draw_push_history()
                self._drawing_stretch_history_pushed = True
            cx = sum(p[0] for p in points) / len(points)
            cy = sum(p[1] for p in points) / len(points)
            self._drawing_stretch_initial_points = points
            self._drawing_stretch_initial_distance = distance
            self._drawing_stretch_centroid = (cx, cy)
            self._drawing_stretch_active = True
            return
        initial_distance = self._drawing_stretch_initial_distance or 0.0
        initial_points = self._drawing_stretch_initial_points
        centroid = self._drawing_stretch_centroid
        if initial_distance <= 0.0 or initial_points is None or centroid is None:
            return
        scale = distance / initial_distance
        scale = max(0.2, min(5.0, scale))
        cx, cy = centroid
        stroke["points"] = [
            (cx + (x - cx) * scale, cy + (y - cy) * scale)
            for x, y in initial_points
        ]
        base_thickness = stroke.get("_base_thickness")
        if base_thickness is None:
            base_thickness = int(stroke.get("thickness", 2))
            stroke["_base_thickness"] = base_thickness
        stroke["thickness"] = int(max(1, round(base_thickness * scale)))
        self._camera_draw_rerender_from_strokes()

    def _update_camera_drawing_canvas(self, frame_shape) -> None:
        if self._drawing_render_target != "camera":
            self._commit_camera_draw_stroke()
            self._camera_draw_last_point = None
            self._camera_draw_erasing = False
            self._release_drawing_grab()
            return
        self._ensure_camera_draw_canvas(frame_shape)
        if self._camera_draw_canvas is None:
            return
        point = self._camera_draw_point(frame_shape)
        if point is None:
            self._commit_camera_draw_stroke()
            self._camera_draw_last_point = None
            self._camera_draw_erasing = False
            self._release_drawing_grab()
            return
        if self._drawing_tool == "grab":
            self._commit_camera_draw_stroke()
            self._camera_draw_erasing = False
            self._camera_draw_last_point = None
            secondary_point = self._drawing_secondary_pinch_point(frame_shape)
            stretch_candidate = False
            if (
                secondary_point is not None
                and self._drawing_grabbed_stroke_index is not None
                and 0 <= self._drawing_grabbed_stroke_index < len(self._camera_draw_strokes)
            ):
                stroke = self._camera_draw_strokes[self._drawing_grabbed_stroke_index]
                radius = max(18.0, float(stroke.get("thickness", 0)) * 1.4 + 8.0)
                stretch_candidate = self._camera_draw_stroke_hits_position(stroke, float(secondary_point[0]), float(secondary_point[1]), radius)
            if stretch_candidate and secondary_point is not None:
                self._drawing_grab_last_point = point
                self._handle_drawing_stretch(point, secondary_point)
                return
            if self._drawing_stretch_active:
                self._release_drawing_stretch()
            self._handle_drawing_grab(point)
            return
        if self._drawing_grabbed_stroke_index is not None:
            self._release_drawing_grab()
        brush = self._drawing_brush_bgr()
        thickness = int(max(2, self._drawing_brush_thickness))
        if self._drawing_tool == "draw":
            self._camera_draw_erasing = False
            if self._drawing_freeze_pen:
                # User has opened the thumb to lift — we're inside the
                # 0.20 s commit hold but still in the grace window so
                # _drawing_tool is "draw". Skip appending any new point
                # this frame so the stroke endpoint doesn't drift along
                # with the natural rightward hand motion that comes
                # with opening the thumb. The previous sample stays
                # as the final point; if the user closes the thumb
                # again before the hold elapses, drawing resumes from
                # there without a stray segment.
                return
            if self._camera_draw_last_point is None:
                # First sample of stroke â€” record only, no draw yet.
                # The Bezier-through-midpoints scheme below needs at
                # least two samples to produce its first segment.
                self._camera_draw_push_history()
                self._camera_draw_last_point = point
                self._camera_draw_active_stroke_points = [(float(point[0]), float(point[1]))]
                return
            # r53 v2: drop samples closer than _DRAWING_MIN_MOVE_PX to
            # the previous point. Kills the residual drift that showed
            # up as micro-strokes when the user paused their hand.
            p_prev = self._camera_draw_last_point
            try:
                min_move = float(self._DRAWING_MIN_MOVE_PX)
                if sys.platform == "darwin":
                    min_move = 6.0
                dx = float(point[0] - p_prev[0])
                dy = float(point[1] - p_prev[1])
                if (dx * dx + dy * dy) < (min_move * min_move):
                    return
            except Exception:
                pass
            color_bgra = (*brush, 255)
            stroke_points = self._camera_draw_active_stroke_points
            if len(stroke_points) < 2:
                # Second sample â€” straight line from P_prev to
                # mid(P_prev, P_new). Re-render produces the same
                # leading stub so canvas matches across redraws.
                mid = (
                    (p_prev[0] + point[0]) // 2,
                    (p_prev[1] + point[1]) // 2,
                )
                cv2.line(self._camera_draw_canvas, p_prev, mid, color_bgra, thickness, cv2.LINE_AA)
            else:
                # Third+ sample â€” quadratic Bezier from
                # mid(P_prev_prev, P_prev) to mid(P_prev, P_new) with
                # P_prev as the control point. This is the standard
                # whiteboard-app smoothing: the curve passes through
                # the midpoints, each raw sample acts as a control,
                # and consecutive segments connect at their shared
                # midpoint anchor with C1 continuity.
                px, py = stroke_points[-2]
                p_prev_prev = (int(round(px)), int(round(py)))
                start = (
                    (p_prev_prev[0] + p_prev[0]) // 2,
                    (p_prev_prev[1] + p_prev[1]) // 2,
                )
                end = (
                    (p_prev[0] + point[0]) // 2,
                    (p_prev[1] + point[1]) // 2,
                )
                arr = self._quad_bezier_polyline(start, p_prev, end)
                cv2.polylines(self._camera_draw_canvas, [arr], False, color_bgra, thickness, cv2.LINE_AA)
            self._camera_draw_active_stroke_points.append((float(point[0]), float(point[1])))
            self._camera_draw_last_point = point
            return
        self._commit_camera_draw_stroke()
        if self._drawing_tool == "erase":
            radius = max(10, int(max(4, self._drawing_eraser_thickness) * 1.35))
            if self._drawing_eraser_mode == "stroke":
                if self._camera_draw_raster_dirty and self._camera_draw_strokes:
                    self._camera_draw_raster_dirty = False
                    self._camera_draw_rerender_from_strokes()
                if not self._camera_draw_erasing:
                    self._camera_draw_push_history()
                    self._camera_draw_erasing = True
                px = float(point[0])
                py = float(point[1])
                hit_index = None
                for idx in range(len(self._camera_draw_strokes) - 1, -1, -1):
                    if self._camera_draw_stroke_hits_position(self._camera_draw_strokes[idx], px, py, float(radius)):
                        hit_index = idx
                        break
                if hit_index is not None:
                    self._camera_draw_strokes.pop(hit_index)
                    self._camera_draw_rerender_from_strokes()
            else:
                if not self._camera_draw_erasing:
                    self._camera_draw_push_history()
                    self._camera_draw_erasing = True
                cv2.circle(self._camera_draw_canvas, point, radius, (0, 0, 0, 0), thickness=-1, lineType=cv2.LINE_AA)
                self._camera_draw_raster_dirty = True
            self._camera_draw_last_point = None
        else:
            self._camera_draw_last_point = None
            self._camera_draw_erasing = False

    def _blend_camera_drawing_overlay(self, frame) -> None:
        if self._drawing_render_target != "camera" or self._camera_draw_canvas is None:
            return
        alpha = self._camera_draw_canvas[:, :, 3:4].astype(np.float32) / 255.0
        if float(alpha.max()) > 0.0:
            overlay_rgb = self._camera_draw_canvas[:, :, :3].astype(np.float32)
            frame[:] = np.clip(frame.astype(np.float32) * (1.0 - alpha) + overlay_rgb * alpha, 0.0, 255.0).astype(np.uint8)
        point = self._camera_draw_point(frame.shape)
        if point is None or self._drawing_tool == "hidden":
            return
        radius = max(6, int(self._drawing_brush_thickness))
        if self._drawing_tool == "draw":
            cv2.circle(frame, point, radius, self._drawing_brush_bgr(), thickness=-1, lineType=cv2.LINE_AA)
            cv2.circle(frame, point, radius + 2, (255, 255, 255), thickness=2, lineType=cv2.LINE_AA)
        elif self._drawing_tool == "erase":
            cv2.circle(frame, point, max(8, int(radius * 1.5)), (255, 255, 255), thickness=2, lineType=cv2.LINE_AA)
        elif self._drawing_tool == "grab":
            grab_color = (30, 220, 255) if self._drawing_grabbed_stroke_index is not None else (200, 200, 200)
            cv2.circle(frame, point, radius + 4, grab_color, thickness=2, lineType=cv2.LINE_AA)
            cv2.circle(frame, point, max(3, radius // 2), grab_color, thickness=-1, lineType=cv2.LINE_AA)
            secondary_point = self._drawing_secondary_pinch_point(frame.shape)
            if secondary_point is not None:
                stretch_color = (255, 140, 30) if self._drawing_stretch_active else (160, 200, 240)
                cv2.circle(frame, secondary_point, radius + 4, stretch_color, thickness=2, lineType=cv2.LINE_AA)
                cv2.circle(frame, secondary_point, max(3, radius // 2), stretch_color, thickness=-1, lineType=cv2.LINE_AA)
                if self._drawing_stretch_active:
                    cv2.line(frame, point, secondary_point, stretch_color, thickness=1, lineType=cv2.LINE_AA)
        else:
            cv2.circle(frame, point, radius, (255, 255, 255), thickness=2, lineType=cv2.LINE_AA)

    def _reset_window_gesture_state(self, *, clear_cooldown: bool = False) -> None:
        self._window_expand_candidate_since = 0.0
        self._window_contract_candidate_since = 0.0
        self._window_close_candidate_since = 0.0
        self._window_pair_smoothed_distance = None
        self._window_pair_last_seen_at = 0.0
        self._window_pair_overlay = None
        self._window_sequence_start_state = None
        self._window_sequence_start_candidate = None
        self._window_sequence_start_candidate_since = 0.0
        self._window_sequence_target_candidate = None
        self._window_sequence_target_candidate_since = 0.0
        if clear_cooldown:
            self._window_gesture_cooldown_until = 0.0

    def _window_pair_pose_metrics(self, hand_reading, now: float | None = None):
        if hand_reading is None:
            return None
        fingers = hand_reading.fingers
        palm_scale = max(float(hand_reading.palm.scale), 1e-6)
        thumb_tip = np.array(hand_reading.landmarks[4][:2], dtype=np.float32)
        index_tip = np.array(hand_reading.landmarks[8][:2], dtype=np.float32)
        raw_distance = float(np.linalg.norm(thumb_tip - index_tip)) / palm_scale
        if self._window_pair_smoothed_distance is None:
            smoothed_distance = raw_distance
        else:
            smoothed_distance = 0.28 * raw_distance + 0.72 * self._window_pair_smoothed_distance
        thumb_ready = (
            fingers["thumb"].state in {"fully_open", "partially_curled", "mostly_curled"}
            and float(fingers["thumb"].openness) >= 0.16
        )
        index_ready = (
            fingers["index"].state in {"fully_open", "partially_curled", "mostly_curled"}
            and float(fingers["index"].openness) >= 0.16
        )
        folded_rest = all(
            fingers[name].state in {"mostly_curled", "closed"}
            or float(getattr(fingers[name], "curl", 0.0) or 0.0) >= 0.40
            or float(getattr(fingers[name], "openness", 0.0) or 0.0) <= 0.56
            for name in ("middle", "ring", "pinky")
        )
        active = bool(thumb_ready and index_ready and folded_rest)
        if active:
            self._window_pair_smoothed_distance = smoothed_distance
            if now is not None:
                self._window_pair_last_seen_at = float(now)
            self._window_pair_overlay = {
                "thumb": (float(thumb_tip[0]), float(thumb_tip[1])),
                "index": (float(index_tip[0]), float(index_tip[1])),
                "raw_distance": raw_distance,
                "distance": smoothed_distance,
            }
            return self._window_pair_overlay
        self._window_pair_smoothed_distance = None
        self._window_pair_overlay = None
        self._window_pair_last_seen_at = 0.0
        return None

    def _window_pair_state(self, distance_ratio: float) -> str:
        value = float(distance_ratio)
        if value <= 0.40:
            return "pinched"
        if value >= 0.54:
            return "apart"
        if 0.30 <= value <= 0.66:
            return "mid"
        return "neutral"

    def _window_close_pose_active(self, hand_reading) -> bool:
        if hand_reading is None:
            return False
        fingers = hand_reading.fingers
        palm_scale = max(float(hand_reading.palm.scale), 1e-6)
        landmarks = hand_reading.landmarks
        thumb_index_ratio = float(np.linalg.norm((landmarks[4] - landmarks[8])[:2])) / palm_scale
        thumb_out = (fingers["thumb"].state == "fully_open" and fingers["thumb"].openness >= 0.70 and fingers["thumb"].palm_distance >= 0.72)
        # Tightened core-fold check: require ALL four fingers to be
        # in 'closed' state (not just 'mostly_curled'). User reported
        # holding the mute (shaka) pose — thumb + pinky out, index/
        # middle/ring curled — eventually triggered close-window
        # because during the hold the pinky drifted from 'open' to
        # 'mostly_curled' even though they didn't release the gesture.
        # 'closed' is stricter and won't accept transient drift.
        folded_core = all(fingers[name].state == "closed" for name in ("index", "middle", "ring", "pinky"))
        # Reject thumbs-up: the thumb tip (landmark 4) must NOT be higher on
        # screen than the middle finger's MCP base joint (landmark 9). Screen
        # Y grows downward, so "higher" means smaller Y. The close-window
        # gesture is meant to be a horizontal / sideways thumb, not a vertical
        # thumbs-up signal the user often makes in unrelated contexts.
        thumb_tip_y = float(landmarks[4][1])
        middle_mcp_y = float(landmarks[9][1])
        thumb_not_above_middle_base = thumb_tip_y >= middle_mcp_y
        return thumb_out and folded_core and thumb_index_ratio >= 0.62 and thumb_not_above_middle_base

    def _handle_window_control_gestures(self, hand_reading, hand_handedness: str | None, now: float) -> bool:
        if hand_handedness != "Right" or hand_reading is None:
            self._reset_window_gesture_state(clear_cooldown=False)
            return False
        if now < self._window_gesture_cooldown_until:
            return False
        controller = self.voice_processor.desktop_controller
        if self._window_close_pose_active(hand_reading):
            if self._window_close_candidate_since <= 0.0:
                self._window_close_candidate_since = now
            if now - self._window_close_candidate_since >= 1.0:
                success = controller.close_active_window()
                self._chrome_control_text = controller.message
                self._spotify_control_text = controller.message
                if success:
                    self.command_detected.emit(controller.message)
                    self._window_gesture_cooldown_until = now + 2.0
                self._reset_window_gesture_state(clear_cooldown=False)
                return True
            return True
        self._window_close_candidate_since = 0.0
        metrics = self._window_pair_pose_metrics(hand_reading, now=now)
        if metrics is None:
            self._window_sequence_start_state = None
            self._window_sequence_start_candidate = None
            self._window_sequence_start_candidate_since = 0.0
            self._window_sequence_target_candidate = None
            self._window_sequence_target_candidate_since = 0.0
            return False
        state = self._window_pair_state(float(metrics["distance"]))
        if self._window_sequence_start_state is None:
            if state not in {"apart", "pinched"}:
                self._window_sequence_start_candidate = None
                self._window_sequence_start_candidate_since = 0.0
                return True
            if state != self._window_sequence_start_candidate:
                self._window_sequence_start_candidate = state
                self._window_sequence_start_candidate_since = now
                return True
            if now - self._window_sequence_start_candidate_since >= 1.3:
                self._window_sequence_start_state = state
                self._window_sequence_target_candidate = None
                self._window_sequence_target_candidate_since = 0.0
                label = "expanded" if state == "apart" else "pinched"
                self._chrome_control_text = f"window control start: {label}"
                self._spotify_control_text = self._chrome_control_text
            return True
        start_state = self._window_sequence_start_state
        target_state = None
        action = None
        if start_state == "apart":
            if state == "pinched":
                target_state = "pinched"
                action = "minimize"
            elif state == "mid":
                target_state = "mid"
                action = "restore"
        elif start_state == "pinched":
            if state == "apart":
                target_state = "apart"
                action = "maximize"
            elif state == "mid":
                target_state = "mid"
                action = "restore"
        if target_state is None:
            self._window_sequence_target_candidate = None
            self._window_sequence_target_candidate_since = 0.0
            return True
        if target_state != self._window_sequence_target_candidate:
            self._window_sequence_target_candidate = target_state
            self._window_sequence_target_candidate_since = now
            return True
        if now - self._window_sequence_target_candidate_since < 1.0:
            return True
        if action == "maximize":
            success = controller.maximize_active_window()
        elif action == "minimize":
            success = controller.minimize_active_window()
        else:
            success = controller.restore_active_window()
        self._chrome_control_text = controller.message
        self._spotify_control_text = controller.message
        if success:
            self.command_detected.emit(controller.message)
            self._window_gesture_cooldown_until = now + 2.0
        self._reset_window_gesture_state(clear_cooldown=False)
        return True

    @staticmethod
    def _gesture_banner_label(prediction) -> tuple[str, bool]:
        # Stable label when non-neutral, else fall back to raw.
        # Matches the original draw_hand_overlay banner logic.
        # "active" (green box) = the chosen label is non-neutral.
        if prediction is None:
            return "", False
        stable = str(getattr(prediction, "stable_label", "neutral") or "neutral")
        raw = str(getattr(prediction, "raw_label", "neutral") or "neutral")
        chosen = stable if stable != "neutral" else raw
        if chosen == "neutral":
            return "", False
        return chosen, True

    def _derive_display_label(self, prediction, hand_reading) -> tuple[str, bool]:
        # Like _gesture_banner_label but promotes the recognizer's
        # base label to its derived variant (three_together,
        # four_together) when the finger pattern matches — so the
        # bbox banner reflects the same label the action-routing
        # layer uses, instead of stalling on "three" / "four" while
        # the engine is firing Chrome / YouTube actions off the
        # derived form. Active flag stays True for any non-neutral
        # chosen label, which keeps the bbox green via the existing
        # color branch.
        if prediction is None:
            return "", False
        if hand_reading is not None:
            try:
                derived = self._derive_app_static_label(prediction, hand_reading)
            except Exception:
                derived = ""
            if derived and derived != "neutral":
                stable = str(getattr(prediction, "stable_label", "neutral") or "neutral")
                if derived != stable:
                    return derived, True
        return self._gesture_banner_label(prediction)

    @staticmethod
    def _filter_banner_label_by_handedness(
        label: str,
        active: bool,
        handedness: Optional[str],
    ) -> tuple[str, bool]:
        # Suppress labels that the static-pose binding registry only
        # recognises on the *other* hand. Without this, the static
        # recognizer happily fires the same shape-based label
        # regardless of which physical hand is in frame, so the user
        # sees confusing banners like "left | mute" on a left hand
        # even though mute is wired exclusively to the right hand
        # (and the action-fire path is already handedness-gated, so
        # the banner is the only visible artifact). Applied to BOTH
        # primary and secondary hands at draw time.
        if not label or label == "neutral":
            return label, active
        if not handedness or handedness not in {"Left", "Right"}:
            return label, active
        own_pose = pose_id_for_static_label(handedness, label)
        if own_pose is not None:
            return label, active
        # Suppress labels not bound on this hand. Covers the
        # other-hand-only case (e.g. mute on left hand — only bound
        # right) and labels with no binding entry at all. Without
        # this the banner showed names the user couldn't use on the
        # active hand, which was visually noisy and misleading.
        return "", False

    # Recognizer labels that are not shipped as named preset
    # gestures. "one" (index finger up) is used internally for
    # voice-listen / drawing / repeat-circle, but there is no
    # user-facing gesture called "one". thumb_up / thumb_down were
    # derived heuristics that collided with snap / fist and are
    # not product gestures — YouTube like/dislike stay on the wheel.
    # Showing any of these on the live viewer implies a preset and
    # collides with custom names.
    _UNNAMED_RECOGNIZER_LABELS = frozenset({"one", "thumb_up", "thumb_down"})

    @classmethod
    def _hide_unnamed_recognizer_label(
        cls, label: str, active: bool,
    ) -> tuple[str, bool]:
        if str(label or "") in cls._UNNAMED_RECOGNIZER_LABELS:
            return "", False
        return label, active

    @staticmethod
    def _build_hand_overlay_info(
        tracked_hand,
        *,
        label: str,
        active: bool,
    ) -> dict:
        # Pack a per-hand display payload for GpuVideoWidget. Tuples
        # / floats only â€” the dict crosses the Qt signal queue so we
        # keep it free of any object that the receiver thread can't
        # safely consume. bbox is normalized [0, 1] image coords.
        landmarks_arr = getattr(tracked_hand, "landmarks", None)
        if landmarks_arr is not None:
            pts = [(float(p[0]), float(p[1])) for p in landmarks_arr]
        else:
            pts = []
        bbox = getattr(tracked_hand, "bbox", None)
        if bbox is not None:
            bbox_tuple = (
                float(bbox.x),
                float(bbox.y),
                float(bbox.width),
                float(bbox.height),
            )
        else:
            bbox_tuple = None
        handedness = str(getattr(tracked_hand, "handedness", "") or "")
        return {
            "landmarks": pts,
            "bbox": bbox_tuple,
            "handedness": handedness,
            "label": str(label or ""),
            "active": bool(active),
        }

    def _normalize_result_right_primary(self, result):
        if not getattr(result, "found", False) or result.tracked_hand is None:
            return result
        primary_label = result.tracked_hand.handedness
        secondary = getattr(result, "secondary_tracked_hand", None)
        sec_label = secondary.handedness if secondary is not None else None
        if primary_label != "Right" and sec_label == "Right":
            return SimpleNamespace(
                found=True,
                frame_index=getattr(result, "frame_index", 0),
                tracked_hand=result.secondary_tracked_hand,
                hand_reading=getattr(result, "secondary_hand_reading", None),
                prediction=getattr(result, "secondary_prediction", None),
                annotated_frame=result.annotated_frame,
                secondary_tracked_hand=result.tracked_hand,
                secondary_hand_reading=result.hand_reading,
                secondary_prediction=result.prediction,
            )
        return result

    def _draw_window_control_overlay(self, frame) -> None:
        overlay = self._window_pair_overlay
        if not overlay or frame is None:
            return
        frame_h, frame_w = frame.shape[:2]
        thumb = overlay.get("thumb")
        index = overlay.get("index")
        if thumb is None or index is None:
            return
        p1 = (int(round(max(0.0, min(1.0, float(thumb[0]))) * max(frame_w - 1, 1))), int(round(max(0.0, min(1.0, float(thumb[1]))) * max(frame_h - 1, 1))))
        p2 = (int(round(max(0.0, min(1.0, float(index[0]))) * max(frame_w - 1, 1))), int(round(max(0.0, min(1.0, float(index[1]))) * max(frame_h - 1, 1))))
        cv2.line(frame, p1, p2, (255, 248, 212), 2, cv2.LINE_AA)
        cv2.circle(frame, p1, 6, (36, 220, 184), thickness=-1, lineType=cv2.LINE_AA)
        cv2.circle(frame, p2, 6, (36, 220, 184), thickness=-1, lineType=cv2.LINE_AA)
        mid_x = int(round((p1[0] + p2[0]) * 0.5))
        mid_y = int(round((p1[1] + p2[1]) * 0.5))
        ratio = float(overlay.get("distance", 0.0) or 0.0)
        cv2.putText(frame, f"{ratio:.2f}", (mid_x + 8, mid_y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (255, 255, 255), 2, cv2.LINE_AA)
    def set_tutorial_context(self, enabled: bool, step_key: str | None = None) -> None:
        self._tutorial_mode_enabled = bool(enabled)
        if enabled:
            normalized_step = str(step_key or "").strip()
            self._tutorial_step_key = normalized_step or None
        else:
            self._tutorial_step_key = None
            # Leaving tutorial entirely lifts any leftover mute block
            # so post-tutorial dictation / normal use can mute again.
            self._tutorial_block_mute = False
        if not self._tutorial_mode_enabled or self._tutorial_step_key != "gesture_wheel":
            self._reset_chrome_wheel()
            self._reset_spotify_wheel()
            self._reset_youtube_wheel()
        if not self._tutorial_mode_enabled:
            self.chrome_router.reset()
            self.spotify_router.reset()

    def _cursor_to_active_monitor(self, x: float, y: float) -> tuple[float, float]:
        """Remap a normalized cursor position [0,1] from "covers the
        full virtual desktop" (the historical meaning of move_normalized)
        to "covers only the user's chosen monitor" when
        config.mouse_active_monitor_index is set. Falls through to
        the input untouched when:
          - The user picked "All Monitors" (index None).
          - QGuiApplication isn't ready yet (very-early-startup edge).
          - The picked monitor index points outside the current
            screens list (display unplugged after Save Locations was
            set).
        Cached cheaply per-call: monitor geometry only changes on
        physical re-plug, but the lookup is microsecond-scale so we
        don't bother with explicit cache invalidation."""
        idx = getattr(self.config, "mouse_active_monitor_index", None)
        if not isinstance(idx, int):
            return (x, y)
        try:
            from PySide6.QtGui import QGuiApplication
            screens = list(QGuiApplication.screens() or [])
            if not (0 <= idx < len(screens)):
                return (x, y)
            mon = screens[idx].geometry()
            # Virtual desktop bounds — fetched from the mouse
            # controller because it goes through the same Win32
            # SM_*VIRTUALSCREEN metrics the actual SetCursorPos
            # call uses, so the rectangles match exactly.
            bounds = self.mouse_controller.virtual_bounds()
            if bounds is None:
                return (x, y)
            vleft, vtop, vw, vh = bounds
            if vw <= 0 or vh <= 0:
                return (x, y)
            # Monitor's region in virtual-desktop normalized coords.
            rel_x = (mon.left() - vleft) / float(vw)
            rel_y = (mon.top() - vtop) / float(vh)
            rel_w = mon.width() / float(vw)
            rel_h = mon.height() / float(vh)
            # User's [0,1] cursor scales into the monitor's slice.
            new_x = rel_x + max(0.0, min(1.0, x)) * rel_w
            new_y = rel_y + max(0.0, min(1.0, y)) * rel_h
            return (max(0.0, min(1.0, new_x)), max(0.0, min(1.0, new_y)))
        except Exception:
            return (x, y)

    def _build_mouse_tracker(self) -> MouseGestureTracker:
        return MouseGestureTracker(
            control_box_center_x=self.config.mouse_control_box_center_x,
            control_box_center_y=self.config.mouse_control_box_center_y,
            control_box_area=self.config.mouse_control_box_area,
            control_box_aspect_power=self.config.mouse_control_box_aspect_power,
        )

    def _blank_mouse_update(self):
        return SimpleNamespace(
            mode_enabled=False,
            cursor_position=None,
            left_click=False,
            left_press=False,
            left_release=False,
            right_click=False,
            scroll_steps=0,
        )

    def apply_config(self, config) -> None:
        self.config = config
        self.volume_overlay.apply_theme(config)
        self.voice_status_overlay.apply_theme(config)
        self.spotify_wheel_overlay.apply_theme(config)
        self.chrome_wheel_overlay.apply_theme(config)
        self.youtube_wheel_overlay.apply_theme(config)
        self.drawing_wheel_overlay.apply_theme(config)
        self.utility_wheel_overlay.apply_theme(config)
        self.mouse_tracker = self._build_mouse_tracker()

    def _refresh_fullscreen_foreground(self, now: float) -> None:
        if (now - self._fullscreen_check_last) < self._FULLSCREEN_POLL_INTERVAL:
            return
        self._fullscreen_check_last = now
        try:
            active = bool(is_foreground_fullscreen())
        except Exception:
            active = False
        process = ""
        if active:
            try:
                info = get_foreground_window_info()
                if info is not None:
                    process = info.process_name or ""
            except Exception:
                process = ""
        was_active = self._fullscreen_foreground_active
        self._fullscreen_foreground_active = active
        self._fullscreen_foreground_process = process
        # Process priority transition. When a fullscreen app
        # (typically a game) grabs the foreground, Windows applies
        # the foreground-boost policy to that process and drops
        # ours -- combined with DWM throttling the GPU compositor
        # of background windows, that's what produces the visible
        # 1-2 s live-view lag the user reported. Boosting our
        # process to ABOVE_NORMAL_PRIORITY_CLASS while fullscreen
        # is up keeps the camera reader + paint threads scheduled
        # promptly. Restored to NORMAL when the game goes away.
        if active and not was_active:
            self._apply_process_priority(above_normal=True)
            # Adaptive GPU (Step 2): fullscreen app just came up.
            # If the user has GPU inference on (config.gpu_mode OR
            # Lite Mode's implicit prefer_gpu=True), swap to CPU
            # MediaPipe so we stop fighting the game for the GPU.
            # Skipped if already suppressed (defensive; shouldn't
            # happen since transitions are gated on was_active).
            if not self._gpu_suppressed_for_fullscreen and self._running:
                wants_gpu = (
                    bool(getattr(self.config, "gpu_mode", False))
                    or bool(getattr(self.config, "lite_mode", False))
                )
                if wants_gpu:
                    self._gpu_suppressed_for_fullscreen = True
                    try:
                        sys.stderr.write(
                            f"[perf-mode] fullscreen '{process or '?'}' detected — "
                            "auto-suppressing GPU inference (swapping to CPU "
                            "MediaPipe) so gesture tracking doesn't fight the "
                            "foreground app for the GPU. Will restore GPU path "
                            "when the fullscreen window closes.\n"
                        )
                        sys.stderr.flush()
                    except Exception:
                        pass
                    try:
                        self._swap_engine_safely()
                    except Exception:
                        pass
        elif was_active and not active:
            self._apply_process_priority(above_normal=False)
            # Adaptive GPU (Step 2): fullscreen app closed / minimized.
            # Restore the GPU inference path if we suppressed it.
            if self._gpu_suppressed_for_fullscreen and self._running:
                self._gpu_suppressed_for_fullscreen = False
                try:
                    sys.stderr.write(
                        "[perf-mode] fullscreen closed — restoring GPU "
                        "inference path (was suppressed while a fullscreen "
                        "foreground app was up).\n"
                    )
                    sys.stderr.flush()
                except Exception:
                    pass
                try:
                    self._swap_engine_safely()
                except Exception:
                    pass

    @staticmethod
    def _apply_process_priority(*, above_normal: bool) -> None:
        import sys
        if sys.platform != "win32":
            return
        try:
            import ctypes
            handle = ctypes.windll.kernel32.GetCurrentProcess()
            # ABOVE_NORMAL_PRIORITY_CLASS = 0x00008000
            # NORMAL_PRIORITY_CLASS       = 0x00000020
            target = 0x00008000 if above_normal else 0x00000020
            ctypes.windll.kernel32.SetPriorityClass(handle, target)
        except Exception:
            pass

    def _build_engine_guarded(self, timeout_seconds: float = 8.0):
        """Build the engine on a helper thread with a hard wall-clock
        timeout, so a DirectML EP init hang (D3D12 adapter enumeration
        blocked by Razer Cortex / Synapse / a laptop's dGPU-iGPU probe)
        cannot freeze the Qt main thread indefinitely.

        v1.1.7 C20: user reported that when gpu_mode was auto-loaded
        from config on startup, the app AND the PC briefly froze during
        engine build. Root cause: HandDetector.__init__ →
        load_hand_runtime → ort.InferenceSession(providers=[DmlEP,...])
        runs on the Qt main thread with no timeout. If DirectML EP init
        blocks (adapter contention), Qt event loop halts.

        Failure recovery: on timeout, force config.gpu_mode=False for
        THIS session (persisted state is left alone so a future launch
        may retry once the contention clears), rebuild on CPU, and emit
        a neutral status message so the user knows GPU didn't engage.
        """
        import concurrent.futures as _cf
        # Log pre-build so a hang shows up as "pre without post" in the
        # log — this is the decisive line the adversarial verifier
        # asked for. Includes the requested engine signature so we can
        # tell whether the hang is in the GPU path specifically.
        try:
            sig = self._compute_engine_signature()
            sys.stderr.write(f"[perf-mode] pre-engine-build sig={sig}\n")
            sys.stderr.flush()
        except Exception:
            pass
        _t0 = time.perf_counter()
        # Try the primary build in a background thread with a hard wait.
        # Pump Qt events during the wait so the UI (splash / status
        # label / mode toggles that are also on the main thread) stays
        # responsive. Poll on a short interval so the response feels
        # snappy on the fast-build success case.
        try:
            from PySide6.QtWidgets import QApplication as _QApp
            _qapp = _QApp.instance()
        except Exception:
            _qapp = None
        exec_ = _cf.ThreadPoolExecutor(max_workers=1, thread_name_prefix="engine-build")
        try:
            fut = exec_.submit(self._build_engine_for_fps_mode)
            engine_new = None
            timed_out = False
            deadline = time.monotonic() + max(0.5, float(timeout_seconds))
            while True:
                try:
                    engine_new = fut.result(timeout=0.05)
                    break
                except _cf.TimeoutError:
                    if time.monotonic() >= deadline:
                        timed_out = True
                        break
                    if _qapp is not None:
                        try:
                            _qapp.processEvents()
                        except Exception:
                            pass
                except Exception:
                    traceback.print_exc()
                    engine_new = None
                    break
        finally:
            exec_.shutdown(wait=False)
        elapsed_ms = (time.perf_counter() - _t0) * 1000.0
        if timed_out:
            try:
                sys.stderr.write(
                    f"[perf-mode] engine build TIMEOUT after {elapsed_ms:.0f}ms "
                    f"— disabling gpu_mode for this session and falling back to CPU\n"
                )
                sys.stderr.flush()
            except Exception:
                pass
            # In-memory disable only. Do NOT persist to disk — user
            # may just be in a transient bad state (Razer Cortex
            # briefly holding the GPU adapter, driver update in
            # progress, etc.). Next launch retries; if it hangs again
            # they see the timeout again and can flip gpu_mode off in
            # Settings.
            try:
                self.config.gpu_mode = False
                self._applied_gpu_mode = False
            except Exception:
                pass
            try:
                self._emit_status(
                    "GPU acceleration didn't start — running on CPU. "
                    "You can retry GPU Mode in Settings."
                )
            except Exception:
                pass
            # Second build attempt on the main thread — GPU is now
            # off in config so this takes the CPU path and is safe
            # to block briefly.
            try:
                engine_new = self._build_engine_for_fps_mode()
            except Exception:
                traceback.print_exc()
                engine_new = None
        try:
            sys.stderr.write(
                f"[perf-mode] post-engine-build elapsed={elapsed_ms:.0f}ms "
                f"timeout={timed_out} ok={engine_new is not None}\n"
            )
            sys.stderr.flush()
        except Exception:
            pass
        return engine_new

    def _compute_engine_signature(self) -> tuple:
        """Full tuple of every input `_build_engine_for_fps_mode` reads.
        Two states with the same signature are guaranteed to produce
        functionally identical engines — so the cache is safe to key on
        this. Any config field that could change the built HandDetector
        MUST be included here or a stale cache hit will occur."""
        low_fps = bool(getattr(self.config, "low_fps_mode", False)) or self._low_fps_auto_engaged
        lite = bool(getattr(self.config, "lite_mode", False))
        gpu = bool(getattr(self.config, "gpu_mode", False))
        mac_boost = False
        fullscreen_suppress = bool(self._gpu_suppressed_for_fullscreen)
        stable_frames_cfg = int(getattr(self.config, "stable_frames_required", 4))
        return (low_fps, lite, gpu, mac_boost, fullscreen_suppress, stable_frames_cfg)

    def _swap_engine_safely(self) -> None:
        # C23: reentrancy guard. _build_engine_guarded pumps Qt events
        # during its future.wait() so the UI stays responsive when
        # DirectML EP init takes 500-1000 ms. That pump also lets OTHER
        # queued Qt signals dispatch — including the second half of a
        # "mode preset" toggle (e.g. clicking a GPU-Mode button
        # actually fires set_lite_mode(False) THEN set_gpu_mode(True),
        # and the second one lands DURING the first swap's build wait).
        # Without this guard, two _swap_engine_safely calls run in
        # parallel, each spawning its own ThreadPoolExecutor + its own
        # ort.InferenceSession(DirectML). The two DirectML session
        # inits contend for the GPU adapter, roughly DOUBLING the total
        # wall time and stalling the paint loop for ~1.5 s at the mode
        # switch (which the user perceives as a 1-2 s delay every time
        # they toggle GPU on). Coalesce: if a swap is already in flight
        # when a new one is requested, record that a re-run is needed
        # and return. When the in-flight swap finishes it re-invokes
        # itself once with the (possibly updated) current config, and
        # the C17 cache HIT branch collapses that second call to a
        # constant-time no-op when the signature didn't change.
        if getattr(self, "_swap_in_progress", False):
            self._swap_pending_re_run = True
            try:
                sys.stderr.write(
                    "[perf-mode] _swap_engine_safely coalesced (another swap "
                    "in flight — will re-check config when it finishes)\n"
                )
                sys.stderr.flush()
            except Exception:
                pass
            return
        self._swap_in_progress = True
        try:
            self._swap_engine_safely_impl()
        finally:
            self._swap_in_progress = False
            if getattr(self, "_swap_pending_re_run", False):
                self._swap_pending_re_run = False
                # Recurse once. If config hasn't changed the C17 cache
                # HIT / "already matches" branch returns immediately
                # (single dict lookup + one signature compute).
                self._swap_engine_safely()

    def _swap_engine_safely_impl(self) -> None:
        # Build the new engine first, then hand it to the runner; the
        # runner's set_engine call acquires the engine lock, blocking
        # briefly until any in-flight inference returns. ONLY THEN is
        # it safe to close() the old engine.
        try:
            lite = bool(getattr(self.config, "lite_mode", False))
            gpu = bool(getattr(self.config, "gpu_mode", False))
            lowfps_cfg = bool(getattr(self.config, "low_fps_mode", False))
            mac_boost = False
            sys.stderr.write(
                f"[perf-mode] _swap_engine_safely START "
                f"lite={lite} gpu={gpu} boost={mac_boost} "
                f"low_fps_cfg={lowfps_cfg} "
                f"auto_engaged={self._low_fps_auto_engaged} "
                f"-> low_fps_active will be {lowfps_cfg or self._low_fps_auto_engaged}\n"
            )
            sys.stderr.flush()
        except Exception:
            pass
        # C17: cache built engines by config signature so repeat mode
        # toggles (Lite ↔ GPU) skip the ~1-2 s ONNX + DirectML session
        # re-init that would otherwise freeze the main thread on every
        # swap. Cache is bounded (~4-6 unique signatures over a session)
        # and cleared in _shutdown_runtime so state doesn't leak between
        # engine restarts.
        swap_start_perf = time.perf_counter()
        if self._engine_cache is None:
            self._engine_cache = {}
        signature = self._compute_engine_signature()
        cached = self._engine_cache.get(signature)
        old_engine = self.engine
        cache_hit = False
        if cached is not None and cached is not old_engine:
            new_engine = cached
            cache_hit = True
            # C17 review fix #10: reset the cached engine's transient
            # state BEFORE handing it to the runner so a mid-hold pose
            # or partially-accumulated dynamic gesture from the last
            # time this engine was active doesn't fire spuriously on
            # the swap-back frame. detector.reset() clears the
            # landmark smoother + last_primary_hand; engine.reset()
            # additionally clears dynamic recognizer history + the
            # stable-label counter.
            try:
                new_engine.reset()
            except Exception:
                pass
        elif cached is old_engine and old_engine is not None:
            # Signature matches the running engine — nothing to do.
            try:
                sys.stderr.write(f"[perf-mode] engine already matches sig={signature}, skipping swap\n")
                sys.stderr.flush()
            except Exception:
                pass
            return
        else:
            # C20: guarded build. On DirectML timeout the helper
            # forces gpu_mode=False and rebuilds on CPU, so
            # new_engine is a valid CPU engine (not None) in the
            # timeout path — cache under the CPU signature not the
            # requested GPU one.
            new_engine = self._build_engine_guarded()
            if new_engine is not None:
                # If the guarded build fell back to CPU, the
                # signature computed inside the helper's post-log is
                # the current one (CPU). Recompute after in case
                # config.gpu_mode was flipped by the fallback.
                self._engine_cache[self._compute_engine_signature()] = new_engine
            else:
                # Build failed with no fallback — keep the current
                # engine to avoid a null-engine session. Skip the
                # runner swap below.
                try:
                    sys.stderr.write(
                        "[perf-mode] engine build returned None — keeping current engine\n"
                    )
                    sys.stderr.flush()
                except Exception:
                    pass
                return
        self._engine_runner.set_engine(new_engine)
        self.engine = new_engine
        # C17 review add (c): wall-clock timing so we can VERIFY the
        # cache actually shortcuts the swap. HIT should be a few ms;
        # MISS is bounded by ONNX + DirectML session build (~1-2 s).
        try:
            elapsed_ms = (time.perf_counter() - swap_start_perf) * 1000.0
            sys.stderr.write(
                f"[perf-mode] engine cache {'HIT' if cache_hit else 'MISS'} "
                f"sig={signature} elapsed={elapsed_ms:.1f}ms "
                f"cache_size={len(self._engine_cache)}\n"
            )
            sys.stderr.flush()
        except Exception:
            pass
        # DO NOT close old_engine — it stays in the cache for reuse on a
        # future swap back to its signature. Closed for real in
        # _shutdown_runtime which drains the cache.

    def _engage_auto_low_fps(self) -> None:
        # Log auto-engage so we can see in the debug log when slow
        # hardware tripped the threshold. Previously this was silent
        # which made it impossible to tell whether dad's PC ever
        # actually engaged the perf pipeline.
        try:
            sys.stderr.write(
                f"[perf-auto] auto-engaging low_fps (current fps={self._fps:.1f}, threshold={self._CRITICAL_FPS_THRESHOLD})\n"
            )
            sys.stderr.flush()
        except Exception:
            pass
        self._low_fps_auto_engaged = True
        self._swap_engine_safely()
        if self._cap is not None:
            self._apply_low_fps_capture_tuning(self._cap)
        # Also reopen camera in ffmpeg-MJPG mode — without this the
        # auto-engage just changes MediaPipe complexity but the camera
        # stays in slow OpenCV-YUY2 mode, so the perceived FPS doesn't
        # improve at all (the camera's hardware rate is now the cap).
        self._apply_perf_camera_path(want_ffmpeg=True)

    def _disengage_auto_low_fps(self) -> None:
        try:
            sys.stderr.write(
                f"[perf-auto] auto-disengaging low_fps (current fps={self._fps:.1f})\n"
            )
            sys.stderr.flush()
        except Exception:
            pass
        self._low_fps_auto_engaged = False
        self._low_fps_below_since = None
        self._low_fps_above_since = None
        self._swap_engine_safely()
        if self._cap is not None and not self._low_fps_active:
            self._restore_normal_capture_tuning(self._cap)
        # Restore OpenCV cap unless another perf mode is still active.
        self._apply_perf_camera_path(want_ffmpeg=self._wants_ffmpeg_cap())

    def _maybe_auto_toggle_low_fps(self, now: float) -> None:
        if getattr(self.config, "low_fps_mode", False):
            return
        fps = self._fps
        if fps <= 0.0:
            return
        # Two-tier auto-engage:
        # 1. FULLSCREEN-GATED tier (existing behavior): when a fullscreen
        #    app has foreground focus and FPS < 18, engage after 4 s. Exit
        #    when FPS >= 18 for 6 s.
        # 2. CRITICAL-FPS tier (new): regardless of fullscreen, when FPS
        #    stays below 12 for 6 s, engage. Targets users on slower
        #    hardware (older iGPUs / GTX 9xx era discrete) who would
        #    otherwise be stuck at 7 fps on the desktop. Uses an
        #    ASYMMETRIC exit threshold (28 fps for 8 s) to prevent thrash
        #    when lite_mode lifts FPS from 7 → ~25 fps but not above the
        #    normal 18 fps gate.
        fullscreen = self._fullscreen_foreground_active
        # Critical-fps threshold (applies regardless of fullscreen).
        if fps < self._CRITICAL_FPS_THRESHOLD:
            self._low_fps_above_since = None
            if self._low_fps_below_since is None:
                self._low_fps_below_since = now
            elif not self._low_fps_auto_engaged and (now - self._low_fps_below_since) >= self._CRITICAL_FPS_ENTER_SECONDS:
                self._engage_auto_low_fps()
            return
        # Fullscreen-gated tier: same shape as before, but only when
        # focus is on a fullscreen app.
        if fullscreen:
            if fps < self._LOW_FPS_AUTO_THRESHOLD:
                self._low_fps_above_since = None
                if self._low_fps_below_since is None:
                    self._low_fps_below_since = now
                elif not self._low_fps_auto_engaged and (now - self._low_fps_below_since) >= self._LOW_FPS_AUTO_ENTER_SECONDS:
                    self._engage_auto_low_fps()
                return
            # FPS above normal threshold + fullscreen — proceed to the
            # exit-condition path below.
        # Exit condition: handle both tiers' "disengage" cases.
        # - If we were auto-engaged via the critical tier, require fps >=
        #   28 for 8 s before disengaging (so a slight bump from lite mode
        #   doesn't thrash).
        # - If we were auto-engaged via the fullscreen tier, require fps
        #   >= 18 for 6 s (existing behavior).
        # - If fps recovered above critical but is below the symmetric
        #   exit threshold and we engaged via the critical path, hold.
        if self._low_fps_auto_engaged:
            # Determine which tier engaged. Use the asymmetric (higher)
            # exit threshold by default for safety — disengaging too
            # eagerly causes thrash; disengaging slowly only costs the
            # user a brief period of suboptimal mode.
            exit_threshold = self._CRITICAL_FPS_EXIT_THRESHOLD if not fullscreen else self._LOW_FPS_AUTO_THRESHOLD
            exit_seconds = self._CRITICAL_FPS_EXIT_SECONDS if not fullscreen else self._LOW_FPS_AUTO_EXIT_SECONDS
            if fps >= exit_threshold:
                if self._low_fps_above_since is None:
                    self._low_fps_above_since = now
                elif (now - self._low_fps_above_since) >= exit_seconds:
                    self._disengage_auto_low_fps()
            else:
                self._low_fps_above_since = None
        else:
            # Not engaged + above critical + not below fullscreen tier:
            # reset both counters so a future dip starts cleanly.
            self._low_fps_below_since = None
            self._low_fps_above_since = None

    def _maybe_offer_low_fps_suggestion(self, now: float) -> None:
        """Track sustained low FPS and show the suggestion overlay when warranted.

        Independent of the fullscreen-gated auto-engage: the suggestion is a
        user-visible offer, not a silent switch, so it should fire on any
        prolonged FPS drop regardless of whether a game is foregrounded.
        Suppressed when low-fps is already on, when we're inside the post-
        dismiss cooldown, or when the toast is already visible.
        """
        if getattr(self.config, "low_fps_mode", False) or self._low_fps_auto_engaged:
            # Already on (user or auto); no reason to suggest.
            self._low_fps_suggest_below_since = None
            return
        if self._low_fps_suggest_visible:
            return
        if now < self._low_fps_suggest_cooldown_until:
            self._low_fps_suggest_below_since = None
            return
        fps = self._fps
        if fps <= 0.0:
            return
        if fps < self._LOW_FPS_SUGGEST_THRESHOLD:
            if self._low_fps_suggest_below_since is None:
                self._low_fps_suggest_below_since = now
            elif (now - self._low_fps_suggest_below_since) >= self._LOW_FPS_SUGGEST_ENTER_SECONDS:
                self._show_low_fps_suggestion()
        else:
            self._low_fps_suggest_below_since = None

    def _show_low_fps_suggestion(self) -> None:
        try:
            self.low_fps_suggestion_overlay.show_suggestion()
            self._low_fps_suggest_visible = True
        except Exception:
            pass

    def _handle_low_fps_suggestion_activate(self) -> None:
        # User clicked "Low FPS Mode" on the toast â€” flip the persistent
        # setting and apply it live. Mirror the path the Settings button uses.
        try:
            self.set_low_fps_mode(True)
        except Exception:
            pass
        try:
            save_config(self.config)
        except Exception:
            pass
        self._low_fps_suggest_below_since = None
        self._low_fps_suggest_cooldown_until = time.time() + self._LOW_FPS_SUGGEST_COOLDOWN_SECONDS

    def _handle_low_fps_suggestion_dismissed(self) -> None:
        self._low_fps_suggest_visible = False
        self._low_fps_suggest_below_since = None
        self._low_fps_suggest_cooldown_until = time.time() + self._LOW_FPS_SUGGEST_COOLDOWN_SECONDS

    def _dismiss_low_fps_suggestion_via_gesture(self) -> None:
        if self._low_fps_suggest_visible:
            try:
                self.low_fps_suggestion_overlay.dismiss()
            except Exception:
                pass

    @staticmethod
    def _empty_frame_skip_inference(had_hand: bool, idle_ticks: int) -> tuple[bool, int]:
        """Decide whether this empty frame can skip MediaPipe.

        Returns (skip, new_idle_ticks). Never skip while a hand is
        tracked so swipe sampling stays full-rate. Empty frames skip
        every other tick so idle GPU cost drops without starving
        reacquisition (bbox / 3→2→1) or collapsing overlay FPS.
        """
        if had_hand:
            return False, 0
        next_ticks = int(idle_ticks) + 1
        return (next_ticks % 2) != 0, next_ticks

    def _note_display_fps(self) -> None:
        """Drive overlay / auto-Low-FPS from camera emit rate.

        Inference-only FPS collapsed to ~6 when empty-frame skip
        avoided `_on_engine_result`, which then auto-engaged Low FPS
        and made 3→2→1 look like repeat_circle.
        """
        now = time.time()
        last = float(getattr(self, "_last_time", 0.0) or 0.0)
        dt = max(now - last, 1e-6) if last > 0.0 else 1e-6
        self._fps = 0.86 * self._fps + 0.14 * (1.0 / dt) if self._fps else (1.0 / dt)
        self._last_time = now

    @staticmethod
    def _builtin_open_hand_swipe_in_flight(prediction) -> bool:
        """True while builtin swipe_left/right is scoring as in-progress.

        Used to park custom SPRING matching during Spotify skip swipes
        so DTW does not hitch the live view on the GUI thread.
        """
        if prediction is None:
            return False
        label = str(getattr(prediction, "dynamic_label", "") or "")
        if label in {"swipe_left", "swipe_right"}:
            return True
        for cand in getattr(prediction, "dynamic_candidates", ()) or ():
            cand_label = str(getattr(cand, "label", "") or "")
            if cand_label not in {"swipe_left", "swipe_right"}:
                continue
            try:
                score = float(getattr(cand, "score", 0.0) or 0.0)
            except Exception:
                score = 0.0
            if score >= 0.45:
                return True
        return False

    def _perf_optimisations_enabled(self) -> bool:
        # Lite Mode, GPU Mode, AND Low FPS Mode (manual or auto-engaged)
        # all imply "the user opted into the performance pipeline" -
        # ffmpeg-MJPG capture, skip-frame inference, throttled debug-
        # frame emit, and the per-frame timing diagnostic. Returning
        # True for ANY of them is what dad's-PC-class hardware needs:
        # auto-engaged low_fps must drop into the same fast pipeline as
        # the manual toggles, otherwise the live camera stays in the
        # 8-10 fps YUY2 ceiling no matter what the engine does.
        return (
            bool(getattr(self.config, "lite_mode", False))
            or bool(getattr(self.config, "gpu_mode", False))
            or self._low_fps_active
        )

    def _any_perf_mode_active(self) -> bool:
        """True if ANY of lite, gpu, low_fps (manual or auto) is on.
        Used by skip-inference / debug-frame throttle / other engine-
        side gates that care about "is any perf mode on."

        NOTE: NO LONGER used to decide the camera-path swap. Use
        _wants_ffmpeg_cap() for that."""
        return (
            bool(getattr(self.config, "lite_mode", False))
            or bool(getattr(self.config, "gpu_mode", False))
            or bool(getattr(self.config, "low_fps_mode", False))
            or self._low_fps_auto_engaged
        )

    def _wants_ffmpeg_cap(self) -> bool:
        """v1.1.7 C24: only GPU Mode uses the ffmpeg-MJPG camera path.
        Lite Mode and Low FPS Mode stay on the default OpenCV cap so
        toggling them doesn't fire a ~2 s camera-swap dance (release
        old cap + time.sleep(0.6) DShow-release + ffmpeg subprocess
        boot + 6-frame warmup discard). Their per-mode benefits — lite
        MediaPipe / smaller inference frame / reduced overlay work /
        skip-inference / lite_paint_mode — don't need ffmpeg's higher-
        fps source to deliver value.

        User's design intent matches: "Lite is a universal CPU boost
        that works without a strong GPU. GPU is the heaviest boost."
        Only the heaviest path warrants the camera swap.

        Ffmpeg reopen still fires exactly ONCE per session per
        Lite/Low-FPS enable/disable cycle — namely when the user
        toggles GPU. First swap into GPU is the last one that pays
        the ~2 s cost; every subsequent Default/Lite/Low-FPS toggle
        is engine-only (~5-10 ms via the C17 cache HIT branch).

        Darwin never uses the ffmpeg-MJPG capture path. Returning True
        here skipped AVFoundation capture tuning (including the 640×480
        Camera Boost reopen) after GPU mode, which made GPU look dead.
        """
        if sys.platform == "darwin":
            return False
        return bool(getattr(self.config, "gpu_mode", False))

    def _preflight_short_shutter_for_ffmpeg(
        self, index: int, device_name: str
    ) -> None:
        """r54 v9: reduce MediaPipe motion blur on GPU-mode ffmpeg cap
        by pre-flighting a short-shutter EXPOSURE write via cv2/DSHOW
        BEFORE ffmpeg claims the filter graph. UVC drivers latch
        exposure at the device level (§4.2.2.1) so the value persists
        across the pin re-open — ffmpeg then reads frames with a
        short shutter (~15 ms) instead of the driver's dim-scene
        long shutter (~40-80 ms), which is what was producing the
        motion blur → landmark jitter the user reported.

        The r49 short-shutter path in `_apply_default_capture_tuning`
        already proved CAP_PROP_EXPOSURE=-6.0 is the property this
        driver class actually latches (2026-07-27 A/B: 12.5 → 32 fps
        on OpenCV). We reuse the same write against the same device
        interface before ffmpeg opens.

        Gating rules:
          - HGR_FFMPEG_SHORT_SHUTTER=0 → off (env opt-out)
          - HGR_FFMPEG_SHORT_SHUTTER=1 → force on (env opt-in)
          - unset → auto-arm only when `classify_camera_shutter_hint`
            says the display_name matches a known generic UVC family
            (Realtek / USB Camera / Sonix / Chicony). Premium cameras
            (Kiyo Pro, Brio, C920, etc.) stay untouched — they have
            their own AE tuning that r50 already respects.
          - Idempotent: skips if this device was already pre-flighted
            in this session (`_ffmpeg_preflight_device` tracks it).

        Not applied when platform is non-Windows (DSHOW only exists
        on Windows). Silent-fails on any exception so a rejecting
        driver never blocks the ffmpeg open path — worst case, we
        skip and fall through to whatever the driver's auto-exposure
        chose (i.e. current behavior).
        """
        if not sys.platform.startswith("win"):
            return
        try:
            env = os.environ.get("HGR_FFMPEG_SHORT_SHUTTER")
        except Exception:
            env = None
        if env == "0":
            return
        # v1.1.7.2 (dad rig 2026-08-19): mirror the OpenCV-cap
        # apply_hint decision logic here. Previously this method only
        # checked the classifier verdict, so a user who had explicitly
        # unchecked "Boost performance for older cameras" (user_chose=True,
        # cfg_ss=False) still got the preflight EXPOSURE=-6 written on
        # ffmpeg-cap open — which then latched at the UVC device level
        # and darkened the OpenCV fallback cap after ffmpeg failed to
        # open at 60 fps. Respect the user's explicit off choice here
        # the same way _apply_default_capture_tuning does.
        try:
            _cfg_ss = bool(getattr(self.config, "camera_force_short_shutter", False))
            _user_chose = bool(getattr(self.config, "camera_force_short_shutter_user_chose", False))
        except Exception:
            _cfg_ss = False
            _user_chose = False
        if env is None:
            try:
                from ..camera.camera_utils import classify_camera_shutter_hint
                verdict = classify_camera_shutter_hint(device_name)
            except Exception:
                verdict = None
            # Explicit user-off overrides everything except env force.
            if _user_chose and not _cfg_ss:
                try:
                    sys.stderr.write(
                        f"[preflight-shutter] skip: user explicitly disabled "
                        f"(user_chose=True, cfg=False) device={device_name!r}\n"
                    )
                    sys.stderr.flush()
                except Exception:
                    pass
                return
            # No env override, no explicit config on → only fire when
            # classifier says generic UVC AND user hasn't opted out.
            if not _cfg_ss and verdict is not True:
                return
            _verdict_label = (
                "config_user_chose_on" if (_user_chose and _cfg_ss)
                else "config" if _cfg_ss
                else "classifier_generic"
            )
        else:
            _verdict_label = "env_force"
        # Idempotence: skip if we already pre-flighted this device
        # this session. Reset in the teardown path (_apply_perf_camera_path
        # want_ffmpeg=False branch, plus _shutdown_runtime).
        if getattr(self, "_ffmpeg_preflight_device", None) == device_name:
            return
        try:
            pre = cv2.VideoCapture(int(index), cv2.CAP_DSHOW)
            if not pre.isOpened():
                try:
                    sys.stderr.write(
                        f"[preflight-shutter] skip: cv2 open failed "
                        f"idx={index} device={device_name!r}\n"
                    )
                    sys.stderr.flush()
                except Exception:
                    pass
                return
            # MJPG hint so the pre-flight cap matches ffmpeg's format
            # request — some drivers only expose the exposure control
            # in their MJPG pin config.
            try:
                pre.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            except Exception:
                pass
            # AUTO_EXPOSURE=0.25 = DSHOW manual. Driver may silently
            # reject this (as r49 documented) but we write it anyway.
            try:
                pre.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
            except Exception:
                pass
            ok_exp = False
            readback_exp = None
            try:
                ok_exp = bool(pre.set(cv2.CAP_PROP_EXPOSURE, -6.0))
                readback_exp = pre.get(cv2.CAP_PROP_EXPOSURE)
            except Exception:
                pass
            try:
                backend = pre.getBackendName()
            except Exception:
                backend = None
            try:
                pre.release()
            except Exception:
                pass
            # Settle window — let the driver commit the write BEFORE
            # ffmpeg's Popen tries to re-acquire the filter graph.
            # Too short → DSHOW handle race (silent-hang retry in
            # open_ffmpeg_cap_with_fps_fallback covers this at ~4 s
            # cost but 150 ms should avoid the retry entirely).
            try:
                time.sleep(0.15)
            except Exception:
                pass
            self._ffmpeg_preflight_device = device_name
            # v1.1.7.9 display-invariance: mark the short-shutter path
            # active so _compensate_short_shutter_for_display lifts the
            # display frame back to natural brightness. The user sees
            # the fps benefit of the shutter cut but never sees the
            # visual dimming caused by 62 ms exposure.
            self._short_shutter_active_for_display = True
            try:
                sys.stderr.write(
                    f"[preflight-shutter] applied device={device_name!r} "
                    f"backend={backend} set_ok_exp={ok_exp} "
                    f"readback_exp={readback_exp} verdict={_verdict_label} "
                    f"env={env!r}\n"
                )
                sys.stderr.flush()
            except Exception:
                pass
        except Exception as exc:
            try:
                sys.stderr.write(
                    f"[preflight-shutter] exception: {type(exc).__name__}: {exc}\n"
                )
                sys.stderr.flush()
            except Exception:
                pass

    def _release_ffmpeg_preflight(self, index: int) -> None:
        """v1.1.7.1 Layer 4: undo a prior preflight EXPOSURE write.

        When `_preflight_short_shutter_for_ffmpeg` fires, it writes
        `CAP_PROP_EXPOSURE=-6` via a temporary cv2/DSHOW cap. UVC drivers
        latch that value at the device level, so releasing the cv2 cap
        does NOT reset the exposure — the next open (ffmpeg or OpenCV)
        inherits the short-shutter state.

        In the happy path (ffmpeg cap opens with bright frames), that
        latch is exactly what we wanted. But when ffmpeg fails after
        preflight — silent hang, luma-net rejection, format error — we
        need to release the latch before falling back to OpenCV, or
        the OpenCV cap opens with the same dark exposure and the user
        sees no improvement over the "just release ffmpeg and open
        OpenCV" fallback path.

        Idempotent: safe to call even if no preflight was applied
        (the cv2 write is silent-fail on rejected drivers). Clears the
        idempotence latch so future ffmpeg opens can re-apply preflight
        for a fresh session/mode.

        Non-Windows: no-op (DSHOW backend is Windows-only).
        """
        if not sys.platform.startswith("win"):
            return
        # Nothing to release if we never preflighted this session.
        if getattr(self, "_ffmpeg_preflight_device", None) is None:
            return
        try:
            pre = cv2.VideoCapture(int(index), cv2.CAP_DSHOW)
            if not pre.isOpened():
                try:
                    pre.release()
                except Exception:
                    pass
                return
            try:
                # v1.1.7.2 (dad rig): nudge EXPOSURE to a bright value
                # FIRST — while the driver is still in manual mode from
                # the earlier preflight write. Some DSHOW drivers only
                # respect EXPOSURE writes while AUTO is manual, so this
                # write goes through cleanly. Then flip AUTO back to
                # auto so the driver can adapt from there.
                pre.set(cv2.CAP_PROP_EXPOSURE, -4.0)
            except Exception:
                pass
            try:
                # DSHOW auto = 0.75. Some drivers also accept 3.0
                # (MSMF/V4L2 convention) — write both so at least one
                # takes on any backend that reaches this code path.
                pre.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)
                pre.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3.0)
            except Exception:
                pass
            try:
                # v1.1.7.2 (dad rig): re-write EXPOSURE=-4 after the
                # AUTO kick, for drivers that ONLY accept EXPOSURE in
                # auto mode. Redundant on most drivers but safe on all.
                pre.set(cv2.CAP_PROP_EXPOSURE, -4.0)
            except Exception:
                pass
            try:
                pre.release()
            except Exception:
                pass
            # Clear the idempotence latch. If the caller retries GPU
            # mode later, preflight can re-arm on a fresh classifier
            # verdict.
            self._ffmpeg_preflight_device = None
            # v1.1.7.9 display-invariance: the driver is now nudged back
            # toward auto-exposure, so the sensor should produce a
            # naturally-bright frame again. Disable the display-side
            # gamma lift so the user sees the true auto-exposure output
            # instead of an over-brightened composite.
            self._short_shutter_active_for_display = False
            try:
                time.sleep(0.15)  # let the driver commit the write
            except Exception:
                pass
            try:
                sys.stderr.write(
                    f"[preflight-shutter] released after ffmpeg fallback: "
                    f"index={index}\n"
                )
                sys.stderr.flush()
            except Exception:
                pass
        except Exception as _exc:
            try:
                sys.stderr.write(
                    f"[preflight-shutter] release exception: "
                    f"{type(_exc).__name__}: {_exc}\n"
                )
                sys.stderr.flush()
            except Exception:
                pass

    def _apply_perf_camera_path(self, *, want_ffmpeg: bool) -> None:
        """Idempotent: reopen the camera in the requested codec path.
        want_ffmpeg=True -> swap to ffmpeg-MJPG dshow capture (breaks
        the OpenCV YUY2 8-30 fps ceiling, requires Windows local USB
        webcam).
        want_ffmpeg=False -> drop back to OpenCV (default).

        Used by set_lite_mode, set_low_fps_mode, set_gpu_mode,
        _engage_auto_low_fps, _disengage_auto_low_fps. Centralises the
        previously-duplicated reopen blocks so a future toggle can't
        accidentally skip the camera path.

        Logs the decision + outcome to stderr so live-fps regressions
        are diagnosable from the log file alone.
        """
        def _log(msg: str) -> None:
            try:
                sys.stderr.write(f"[perf-camera] {msg}\n")
                sys.stderr.flush()
            except Exception:
                pass

        if not self._running or self._cap is None:
            _log(f"skipped: running={self._running} cap_set={self._cap is not None}")
            return
        info = self._camera_info
        if info is None:
            _log("skipped: no camera_info")
            return
        index = getattr(info, "index", None)
        if index is None or int(index) < 0:
            _log(f"skipped: index={index!r} (phone-camera path)")
            return
        if not sys.platform.startswith("win"):
            _log("skipped: non-Windows")
            return
        currently_ffmpeg = "FfmpegMjpegCapture" in type(self._cap).__name__
        if want_ffmpeg and currently_ffmpeg:
            _log("noop: already on ffmpeg-MJPG")
            return
        if not want_ffmpeg and not currently_ffmpeg:
            _log("noop: already on OpenCV")
            return
        old_cap = self._cap
        self._cap = None
        try:
            old_cap.release()
        except Exception:
            pass
        if want_ffmpeg:
            # DSHOW handle release is async on Windows. When we call
            # OpenCV's release() above and immediately launch ffmpeg,
            # the camera's DirectShow filter graph hasn't finished
            # tearing down yet, so ffmpeg's `-i video=...` blocks on
            # the driver and looks like a silent hang (proc_alive=True,
            # stderr empty, no frames). The hang masks itself as
            # "60 fps unsupported by camera" and we silently fall back
            # to 30 fps. A short pause lets the driver actually release
            # the handle so ffmpeg gets a clean open at the requested
            # rate. 600 ms is empirically enough for Razer Kiyo Pro,
            # Logitech BRIO, EOS Webcam Utility; cheaper webcams
            # release in <200 ms but the extra latency is invisible
            # next to the camera-open warmup that already happens.
            try:
                time.sleep(0.6)
            except Exception:
                pass
            from ..camera.ffmpeg_capture import (
                FfmpegMjpegCapture,  # noqa: F401 - keep import for currently_ffmpeg check
                resolve_dshow_device_for_index,
            )
            device_name = resolve_dshow_device_for_index(
                int(index),
                qt_name_hint=str(getattr(info, "display_name", "") or ""),
            )
            if not device_name:
                _log("ffmpeg path: resolve_dshow_device_for_index returned empty - falling back to OpenCV")
                recovered = open_camera_by_index(int(index), max_index=self.config.camera_scan_limit)
                if isinstance(recovered, tuple) and len(recovered) >= 2 and recovered[1] is not None:
                    self._cap = recovered[1]
                return
            # C27: open the ffmpeg-MJPG cap at 640x480, not 1280x720.
            # The higher resolution ~3× the bytes per frame through
            # every pipeline stage (ffmpeg pipe read, reader thread
            # decode, QImage.copy in the widget). On the user's Kiyo
            # Pro, that resolution difference produced a persistent
            # 1-2 second visual lag in GPU Mode that wasn't present
            # in Default (which uses 640x480 OpenCV). User said
            # utilization and fps are fine — priority is latency —
            # so match Default's resolution. ONNX inference still
            # runs at max_process_width=960 (from _NORMAL_PROCESS_WIDTH)
            # which is now a no-op resize on 640×480 input; hand
            # tracking accuracy is essentially the same on a
            # face-and-hand-in-frame use case at either resolution.
            # r54 v9: pre-flight short shutter via cv2/DSHOW so ffmpeg
            # inherits a fast-shutter driver state → less motion blur
            # → smoother MediaPipe landmarks. Gated by device
            # classifier so premium cams stay untouched. See helper
            # docstring for the full logic + env-var opt-out.
            self._preflight_short_shutter_for_ffmpeg(int(index), device_name)
            # v1.1.7.1: luma_min_threshold=50 is the auto-recovery net
            # for cameras whose driver cuts shutter too aggressively at
            # the requested fps and produces a frame MediaPipe can't
            # detect hands in. See open_ffmpeg_cap_with_fps_fallback's
            # docstring for the full failure mode + rationale. Kicks
            # in for HP HD Camera / cheap laptop webcams that can't
            # sustain 60 fps in indoor lighting; Kiyo Pro / Brio /
            # C920 and healthy driver states pass the threshold on
            # first try and pay only the ~0.4 s sample-window cost.
            ffmpeg_cap = open_ffmpeg_cap_with_fps_fallback(
                device_name, width=640, height=480,
                luma_min_threshold=50.0,
            )
            if ffmpeg_cap is not None and ffmpeg_cap.isOpened():
                self._cap = ffmpeg_cap
                _log(f"engaged ffmpeg-MJPG cap for device={device_name!r}")
                return
            if ffmpeg_cap is not None:
                try:
                    ffmpeg_cap.release()
                except Exception:
                    pass
            _log(f"ffmpeg cap failed for device={device_name!r}, falling back to OpenCV")
            # v1.1.7.1 Layer 4: if we pre-flighted the shutter and the
            # ffmpeg cap ended up unusable (silent hang, luma-net
            # exhaustion, format rejection), the EXPOSURE=-6 write is
            # still latched at the UVC device level. The OpenCV cap
            # we're about to open would inherit that dark state and
            # look identical to the ffmpeg cap we just released.
            # Release the preflight by writing DSHOW-auto back — worst
            # case (driver rejects) it's a harmless no-op; best case
            # the OpenCV cap opens with the natural auto-exposure.
            try:
                self._release_ffmpeg_preflight(int(index))
            except Exception:
                pass
            recovered = open_camera_by_index(int(index), max_index=self.config.camera_scan_limit)
            if isinstance(recovered, tuple) and len(recovered) >= 2 and recovered[1] is not None:
                self._cap = recovered[1]
                # v1.1.7.2 (dad rig): the OpenCV fallback cap needs
                # `_apply_default_capture_tuning` to run so the standard
                # decision logic (should_kick, luma verify, auto-rollback)
                # gets a chance to correct any lingering manual state
                # the preflight release couldn't undo. Without this the
                # OpenCV cap inherits whatever state the driver latched
                # during the failed ffmpeg attempt.
                try:
                    self._apply_default_capture_tuning(recovered)
                except Exception:
                    pass
        else:
            # r54 v9: leaving ffmpeg mode → clear the pre-flight
            # idempotence latch so the next re-entry to ffmpeg mode
            # (or a fresh device) re-arms the short-shutter write.
            self._ffmpeg_preflight_device = None
            # v1.1.7.2 (dad rig): also release any latched preflight
            # EXPOSURE state on the driver before we open OpenCV.
            # Without this, dropping from GPU/Lite/LowFps → Default
            # (any transition that pushed us into ffmpeg mode earlier)
            # leaves the driver in short-shutter mode and the fresh
            # OpenCV cap opens dark.
            try:
                self._release_ffmpeg_preflight(int(index))
            except Exception:
                pass
            recovered = open_camera_by_index(int(index), max_index=self.config.camera_scan_limit)
            if isinstance(recovered, tuple) and len(recovered) >= 2 and recovered[1] is not None:
                self._cap = recovered[1]
                _log("restored OpenCV cap")
                try:
                    self._apply_default_capture_tuning(recovered)
                except Exception:
                    pass

    def _build_engine_for_fps_mode(self) -> GestureRecognitionEngine:
        self._low_fps_active = bool(getattr(self.config, "low_fps_mode", False)) or self._low_fps_auto_engaged
        lite_active = bool(getattr(self.config, "lite_mode", False))
        mac_boost = False
        # GPU Mode threads through to every detector flavour. The
        # runtime loader honours it best-effort and falls back to
        # CPU MediaPipe when no GPU path is reachable, so toggling
        # it on can never break gesture recognition â€” it just won't
        # speed anything up if the GPU path can't engage.
        prefer_gpu = bool(getattr(self.config, "gpu_mode", False))
        # v1.1.7 Step 2 (adaptive GPU): if a fullscreen foreground app
        # (game) is up and we auto-suppressed the GPU path, force
        # prefer_gpu=False for this rebuild. When the game closes the
        # transition-detector flips the flag back and swaps the engine
        # again — no config change, fully automatic.
        if self._gpu_suppressed_for_fullscreen:
            prefer_gpu = False
        if self._low_fps_active:
            # Low-FPS already implies lite landmark model â€” keep its
            # tuned thresholds; lite_mode would be redundant here.
            detector = HandDetector(
                min_detection_confidence=0.34,
                min_tracking_confidence=0.22,
                model_complexity=0,
                miss_tolerance_seconds=0.24,
                max_process_width=self._LOW_FPS_PROCESS_WIDTH,
                smoother=AdaptiveLandmarkSmoother(alpha=0.66, min_alpha=0.24, max_alpha=0.88),
                secondary_smoother=AdaptiveLandmarkSmoother(alpha=0.66, min_alpha=0.24, max_alpha=0.88),
                prefer_gpu=prefer_gpu,
            )
            stable_frames = 1
        elif sys.platform == "darwin" and (lite_active or prefer_gpu):
            # Mac ladder: Lite = 720p + 384-px CPU; GPU = 640×480 +
            # 480-px CoreML. Performance Boost was dropped — FaceTime
            # is 30 fps so a third 256-px mode did not raise actual fps.
            if lite_active:
                prefer_gpu = False
            process_width = (
                self._MAC_GPU_PROCESS_WIDTH
                if prefer_gpu
                else self._LITE_MODE_PROCESS_WIDTH
            )
            detector = HandDetector(
                model_complexity=0,
                max_process_width=process_width,
                prefer_gpu=prefer_gpu,
            )
            stable_frames = 1
            try:
                sys.stderr.write(
                    f"[perf-mode] darwin engine lite={lite_active} "
                    f"gpu={prefer_gpu} boost={mac_boost} "
                    f"process_width={process_width}\n"
                )
                sys.stderr.flush()
            except Exception:
                pass
        elif lite_active:
            # Lite Mode (v1.1.7 C23): CPU-only. The previous version
            # silently set prefer_gpu=True so Lite would use ONNX +
            # DirectML on GPU-capable hardware — but that made two
            # things worse:
            #   1. Lite and GPU Mode became functionally identical on
            #      the common case (both landed on DirectML with only
            #      the inference-input width differing), so users saw
            #      no meaningful difference when toggling between them.
            #   2. Every switch INTO Lite paid the 500-1000 ms
            #      ort.InferenceSession(DirectML) init on the Qt main
            #      thread. The user perceived that as a 1-2 second
            #      lag every time they toggled Lite. Confirmed via the
            #      [perf-mode] engine-build elapsed=795ms log line.
            # Now Lite = lite landmark model + narrower inference frame
            # + CPU MediaPipe. Engine build is ~5 ms (MediaPipe init is
            # cheap). Users who want the GPU acceleration should use
            # GPU Mode explicitly. Matches user's stated intent for the
            # mode topology: "Lite is a universal CPU-cost boost that
            # works without a strong GPU."
            detector = HandDetector(
                model_complexity=0,
                max_process_width=self._LITE_MODE_PROCESS_WIDTH,
                prefer_gpu=False,
            )
            # C26: user reported that Lite Mode felt identical to
            # Default on their strong hardware (camera Synapse-capped
            # at 35 fps, so no fps or CPU delta was perceivable). The
            # workflow diagnosed that Lite's real per-frame benefits
            # are below sample noise when the camera is the bottleneck.
            # Give Lite ONE lever the user WILL feel on every camera:
            # gesture-snap. stable_frames=1 fires a static gesture on
            # the first consistent frame instead of waiting for the
            # default half-of-stable_frames_required (typically 2-3
            # frames = ~60-90 ms). Matches the Low FPS branch's
            # precedent above.
            #
            # Tradeoff: rapid pose transitions (e.g. moving fist →
            # open_palm passes through four/three) can now fire a
            # phantom intermediate gesture. The user opted into Lite
            # explicitly, so this is acceptable — and the whole
            # point of the mode is to be a "snappier response"
            # tradeoff. If false-positives become a support issue,
            # gate this behind a per-user config knob.
            stable_frames = 1
        else:
            # r54 v8: C31 baseline + tighter landmark smoother. User
            # felt residual skeleton jitter on a still hand after the
            # C31 revert and explicitly opted into deviating from the
            # baseline for reduced landmark noise. Smoother params
            # tightened from C31 defaults (0.58 / 0.18 / 0.76) to
            # (0.50 / 0.15 / 0.62): ~20 % less per-frame jitter, ~15 ms
            # extra lag on fast motion. Baseline memory updated.
            detector = HandDetector(
                max_process_width=self._NORMAL_PROCESS_WIDTH,
                prefer_gpu=prefer_gpu,
                smoother=AdaptiveLandmarkSmoother(
                    alpha=0.50, min_alpha=0.15, max_alpha=0.62,
                ),
                secondary_smoother=AdaptiveLandmarkSmoother(
                    alpha=0.50, min_alpha=0.15, max_alpha=0.62,
                ),
            )
            stable_frames = max(2, self.config.stable_frames_required // 2)
        return GestureRecognitionEngine(
            detector=detector,
            stable_frames_required=stable_frames,
            low_fps_mode=self._low_fps_active,
        )

    def set_low_fps_mode(self, enabled: bool) -> None:
        enabled = bool(enabled)
        self.config.low_fps_mode = enabled
        prev_applied = self._applied_low_fps_mode
        try:
            sys.stderr.write(
                f"[perf-mode] set_low_fps_mode({enabled}) running={self._running} "
                f"applied_was={prev_applied} -> applying={prev_applied != enabled}\n"
            )
            sys.stderr.flush()
        except Exception:
            pass
        if not enabled:
            self._low_fps_auto_engaged = False
            self._low_fps_below_since = None
            self._low_fps_above_since = None
        self._low_fps_last_process = 0.0
        if prev_applied == enabled:
            return
        self._applied_low_fps_mode = enabled
        if self._running:
            self._swap_engine_safely()
            self._fps = 0.0
            if self._cap is not None:
                if self._low_fps_active:
                    self._apply_low_fps_capture_tuning(self._cap)
                else:
                    self._restore_normal_capture_tuning(self._cap)
            # Critical: reopen camera with ffmpeg-MJPG when ANY perf
            # mode is active. Previously set_low_fps_mode only tuned
            # the existing OpenCV cap, which means a user enabling
            # Low FPS Mode on slow hardware stayed on the YUY2 8-10
            # fps ceiling and the mode produced no noticeable effect.
            self._apply_perf_camera_path(want_ffmpeg=self._wants_ffmpeg_cap())
            # C25: also refresh the OpenCV cap fps request so a Low
            # FPS → Lite transition picks up Lite's 60 fps target
            # (the _restore_normal_capture_tuning above uses 30 fps).
            try:
                self._apply_default_capture_tuning((None, self._cap))
            except Exception:
                pass

    def _darwin_capture_wh(self) -> tuple:
        """AVFoundation size to request on the next Mac camera open.

        Mid-session cap.set(WIDTH/HEIGHT) is ignored or renegotiates
        into a worse ~11 fps mode. Callers that need a new size must
        reopen via `_reopen_macos_camera`. Lite keeps 720p.
        """
        if sys.platform != "darwin":
            return (1280, 720)
        if (
            bool(getattr(self.config, "gpu_mode", False))
            or bool(getattr(self.config, "camera_force_short_shutter", False))
        ):
            return (640, 480)
        return (1280, 720)

    def _darwin_open_kwargs(self) -> dict:
        if sys.platform != "darwin":
            return {}
        width, height = self._darwin_capture_wh()
        return {"capture_width": width, "capture_height": height}

    def _macos_capture_size_matches(self, width: int, height: int) -> bool:
        cap = self._cap
        if cap is None:
            return False
        try:
            current_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            current_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        except Exception:
            return False
        return abs(current_w - int(width)) <= 16 and abs(current_h - int(height)) <= 16

    def _reopen_macos_camera(self) -> None:
        """Release and reopen the AVFoundation cap at the desired size.

        No-op on Windows. Logs the actual negotiated size so a remote
        Mac test can confirm the mode took effect.
        """
        if sys.platform != "darwin" or not self._running:
            return
        info = self._camera_info
        index = getattr(info, "index", None) if info is not None else None
        if index is None or int(index) < 0:
            return
        width, height = self._darwin_capture_wh()
        if self._macos_capture_size_matches(width, height):
            try:
                sys.stderr.write(
                    f"[perf-camera] mac reopen skipped: already {width}x{height}\n"
                )
                sys.stderr.flush()
            except Exception:
                pass
            try:
                self._apply_default_capture_tuning((info, self._cap))
            except Exception:
                pass
            return
        old_cap = self._cap
        self._cap = None
        if old_cap is not None:
            try:
                old_cap.release()
            except Exception:
                pass
        recovered = open_camera_by_index(
            int(index),
            max_index=self.config.camera_scan_limit,
            capture_width=width,
            capture_height=height,
        )
        if isinstance(recovered, tuple) and len(recovered) >= 2 and recovered[1] is not None:
            self._cap = recovered[1]
            if recovered[0] is not None:
                self._camera_info = recovered[0]
            try:
                self._apply_default_capture_tuning(recovered)
            except Exception:
                pass
            try:
                actual_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
                actual_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
                sys.stderr.write(
                    f"[perf-camera] mac reopened requested={width}x{height} "
                    f"actual={actual_w}x{actual_h}\n"
                )
                sys.stderr.flush()
            except Exception:
                pass
            return
        try:
            sys.stderr.write(
                f"[perf-camera] mac reopen failed for {width}x{height}; "
                "leaving cap unset for recovery\n"
            )
            sys.stderr.flush()
        except Exception:
            pass

    def set_lite_mode(self, enabled: bool) -> None:
        # Lite-model toggle. Compares against LAST APPLIED state, not
        # config — the UI updates config.lite_mode before calling this,
        # so config-based comparison always shows no-op. See the
        # _applied_*_mode docstring in __init__.
        enabled = bool(enabled)
        self.config.lite_mode = enabled
        prev_applied = self._applied_lite_mode
        try:
            sys.stderr.write(
                f"[perf-mode] set_lite_mode({enabled}) running={self._running} "
                f"applied_was={prev_applied} -> applying={prev_applied != enabled}\n"
            )
            sys.stderr.flush()
        except Exception:
            pass
        if not self._running:
            return
        if prev_applied == enabled:
            return
        self._applied_lite_mode = enabled
        self._swap_engine_safely()
        self._fps = 0.0
        if sys.platform == "darwin":
            # Do not rewrite CAP_PROP_FPS/size on a live AVFoundation
            # session — that renegotiation is what dropped Lite to ~11 fps.
            return
        self._apply_perf_camera_path(want_ffmpeg=self._wants_ffmpeg_cap())
        # C25: re-apply OpenCV capture tuning when a mode toggle
        # changes the target fps request. Noop when the current cap
        # is our FfmpegMjpegCapture wrapper. Handles the case where
        # user toggles Lite ON while the camera path is a noop
        # (Default ↔ Lite share the OpenCV cap under C24), so
        # without this the driver stays on the pre-toggle fps
        # request and Lite doesn't get its 60-fps bump.
        try:
            self._apply_default_capture_tuning((None, self._cap))
        except Exception:
            pass

    def set_gpu_mode(self, enabled: bool) -> None:
        # GPU-acceleration toggle. See set_lite_mode for the
        # config-vs-applied rationale.
        enabled = bool(enabled)
        self.config.gpu_mode = enabled
        prev_applied = self._applied_gpu_mode
        try:
            sys.stderr.write(
                f"[perf-mode] set_gpu_mode({enabled}) running={self._running} "
                f"applied_was={prev_applied} -> applying={prev_applied != enabled}\n"
            )
            sys.stderr.flush()
        except Exception:
            pass
        if not self._running:
            return
        if prev_applied == enabled:
            return
        self._applied_gpu_mode = enabled
        self._swap_engine_safely()
        self._fps = 0.0
        if sys.platform == "darwin":
            self._reopen_macos_camera()
            return
        self._apply_perf_camera_path(want_ffmpeg=self._wants_ffmpeg_cap())
        # C25: re-apply OpenCV capture tuning when a mode toggle
        # changes the target fps request. Noop when the current cap
        # is our FfmpegMjpegCapture wrapper. Handles the case where
        # user toggles Lite ON while the camera path is a noop
        # (Default ↔ Lite share the OpenCV cap under C24), so
        # without this the driver stays on the pre-toggle fps
        # request and Lite doesn't get its 60-fps bump.
        try:
            self._apply_default_capture_tuning((None, self._cap))
        except Exception:
            pass

    def set_mac_performance_boost(self, enabled: bool) -> None:
        """Removed from the Mac UI. Kept as a no-op so old settings.json
        and leftover Save-Changes keys cannot re-enable it."""
        self.config.mac_performance_boost = False
        if sys.platform != "darwin":
            return
        prev_applied = self._applied_mac_performance_boost
        if prev_applied is False:
            return
        self._applied_mac_performance_boost = False
        if not self._running:
            return
        if prev_applied is True:
            self._swap_engine_safely()
            self._fps = 0.0
            self._reopen_macos_camera()

    def set_force_ten_fps_test_mode(self, enabled: bool) -> None:
        self.config.force_ten_fps_test_mode = bool(enabled)
        self._low_fps_last_process = 0.0

    def _maybe_log_perf_monitor(self) -> None:
        """C22: log a system-resource snapshot every 5 s so the user
        can see CPU / GPU / RAM change as they toggle modes from the
        log alone. Non-blocking: nvidia-smi is queried on a background
        thread and the result cached across ticks, so a slow query
        never blocks the main-thread tick loop.

        Emits ONE stderr line every ~5 s in the shape:

            [perf-monitor] proc: cpu=X.X% rss=NNMB | sys: cpu=X.X% |
                gpu: NN% mem=NN%/NNNMB (RTX 4070) | tick_fps=NN
                emit_fps=NN

        Any component that can't be measured is elided from the line
        (e.g. non-NVIDIA systems drop the gpu block entirely).
        """
        try:
            _now = time.monotonic()
            if self._perf_monitor_last_log <= 0.0:
                self._perf_monitor_last_log = _now
                # Kick off the first background GPU query so the log
                # has data by the time we hit the 5-second window.
                self._perf_monitor_kickoff_gpu_query()
                return
            if (_now - self._perf_monitor_last_log) < 5.0:
                return
            self._perf_monitor_last_log = _now
            parts: list[str] = []
            # ---- Process CPU + RAM via psutil ----
            if self._perf_monitor_proc is not None:
                try:
                    p_cpu = self._perf_monitor_proc.cpu_percent(interval=None)
                    p_rss_mb = self._perf_monitor_proc.memory_info().rss / (1024 * 1024)
                    parts.append(f"proc: cpu={p_cpu:.1f}% rss={p_rss_mb:.0f}MB")
                except Exception:
                    pass
            # ---- System-wide CPU ----
            if self._perf_monitor_psutil is not None:
                try:
                    sys_cpu = self._perf_monitor_psutil.cpu_percent(interval=None)
                    parts.append(f"sys: cpu={sys_cpu:.1f}%")
                except Exception:
                    pass
            # ---- GPU: consume the background-thread result if ready,
            # then kick off the next query. This keeps the main tick
            # off nvidia-smi's ~50-100 ms subprocess spawn.
            gpu_row = self._perf_monitor_consume_gpu_result()
            if gpu_row is not None:
                util_pct, mem_used_mb, mem_total_mb, gpu_name = gpu_row
                mem_pct = (100.0 * mem_used_mb / mem_total_mb) if mem_total_mb > 0 else 0.0
                parts.append(
                    f"gpu: {util_pct:.0f}% mem={mem_pct:.0f}%/{mem_used_mb:.0f}MB "
                    f"({gpu_name})"
                )
            self._perf_monitor_kickoff_gpu_query()
            # ---- fps context ----
            try:
                parts.append(f"self._fps={self._fps:.1f}")
            except Exception:
                pass
            try:
                sys.stderr.write("[perf-monitor] " + " | ".join(parts) + "\n")
                sys.stderr.flush()
            except Exception:
                pass
            # r50: one-shot post-hint fps breadcrumb. Fires once, on the
            # first perf-monitor tick that lands >=3 s after the r49
            # hint was armed, so support readers can correlate the
            # hint's outcome against delivered fps without a bespoke
            # timer. Skipped when the wall-clock anchor is absent
            # (hint never applied) or already reported.
            try:
                _armed_ts = getattr(self, "_r49_hint_applied_wallclock", None)
                _reported = getattr(self, "_r50_post_hint_fps_reported", False)
                if _armed_ts is not None and not _reported:
                    import time as _t
                    elapsed = _t.time() - float(_armed_ts)
                    if elapsed >= 3.0:
                        _reason = getattr(self, "_r50_last_decision_reason", "unknown")
                        _display = getattr(self, "_r50_last_display_name", "")
                        sys.stderr.write(
                            f"[r49-short-shutter] post_hint_fps={self._fps:.1f} "
                            f"elapsed={elapsed:.1f}s reason={_reason} "
                            f"display_name={_display!r}\n"
                        )
                        sys.stderr.flush()
                        self._r50_post_hint_fps_reported = True
            except Exception:
                pass
        except Exception:
            # Never let the diagnostic take down the tick loop.
            pass

    def _perf_monitor_kickoff_gpu_query(self) -> None:
        """Spawn nvidia-smi in a helper thread. Result is stashed on
        `_perf_monitor_gpu_result` for the next tick that reaches the
        5-second window. If nvidia-smi is missing or the machine has no
        NVIDIA GPU, the result stays None forever (silent no-op)."""
        if getattr(self, "_perf_monitor_gpu_query_in_flight", False):
            return
        self._perf_monitor_gpu_query_in_flight = True
        def _worker() -> None:
            try:
                import subprocess as _sp
                out = _sp.check_output(
                    [
                        "nvidia-smi",
                        "--query-gpu=utilization.gpu,memory.used,memory.total,name",
                        "--format=csv,noheader,nounits",
                    ],
                    stderr=_sp.DEVNULL,
                    timeout=1.5,
                    creationflags=getattr(_sp, "CREATE_NO_WINDOW", 0),
                )
                text = out.decode("utf-8", errors="replace").strip().splitlines()
                if not text:
                    return
                first_row = text[0]
                fields = [f.strip() for f in first_row.split(",")]
                if len(fields) < 4:
                    return
                util_pct = float(fields[0])
                mem_used_mb = float(fields[1])
                mem_total_mb = float(fields[2])
                gpu_name = fields[3]
                self._perf_monitor_gpu_result = (util_pct, mem_used_mb, mem_total_mb, gpu_name)
            except Exception:
                # nvidia-smi missing, timeout, non-NVIDIA GPU, etc.
                pass
            finally:
                self._perf_monitor_gpu_query_in_flight = False
        try:
            threading.Thread(target=_worker, name="perf-monitor-gpu", daemon=True).start()
        except Exception:
            self._perf_monitor_gpu_query_in_flight = False

    def _perf_monitor_consume_gpu_result(self):
        """Return the most-recent nvidia-smi row (util, used, total,
        name) or None if we haven't gotten one yet."""
        return getattr(self, "_perf_monitor_gpu_result", None)

    def _should_skip_forced_fps_tick(self, now: float) -> bool:
        if not bool(getattr(self.config, "force_ten_fps_test_mode", False)):
            self._low_fps_last_process = 0.0
            return False
        target_fps = max(1.0, float(self._FORCED_TEST_FPS_TARGET))
        min_interval = 1.0 / target_fps
        if self._low_fps_last_process <= 0.0:
            self._low_fps_last_process = now
            return False
        if (now - self._low_fps_last_process) < min_interval:
            return True
        self._low_fps_last_process = now
        return False

    def start(self) -> None:
        if self._running:
            return
        self._shutdown_runtime(emit_signal=False)
        # C20: guard the initial build with a wall-clock timeout so a
        # persisted-from-config gpu_mode=True that hangs on DirectML
        # init cannot freeze the Qt main thread at startup. Falls back
        # to CPU on timeout with a neutral status message.
        self.engine = self._build_engine_guarded()
        # C17: seed the engine cache with the startup engine so a later
        # toggle back to this signature hits the cache instead of paying
        # another ONNX + DirectML session build.
        if self._engine_cache is None:
            self._engine_cache = {}
        if self.engine is not None:
            self._engine_cache[self._compute_engine_signature()] = self.engine
        # Spin up the background-thread engine runner. The lambda
        # bridge translates the runner-thread callback into a
        # cross-thread Qt signal emit, which auto-queues the result
        # delivery onto the GUI thread.
        self._engine_runner.set_engine(self.engine)
        if not self._engine_runner.is_running:
            self._engine_runner.start(
                result_callback=lambda f, r: self._engine_result_ready.emit(f, r)
            )
        self._volume_message = self.volume_controller.message
        self._volume_level = self.volume_controller.refresh_cache().level_scalar
        self._volume_mode_active = False
        self._volume_status_text = "idle"
        self._volume_muted = self._read_system_mute()
        self._volume_overlay_visible = False
        self._mute_block_until = 0.0
        self.volume_tracker.reset(self._volume_level, self._volume_muted)
        self.volume_overlay.hide_overlay()
        self.voice_status_overlay.hide_overlay()
        self.mouse_controller.release_all()
        self.mouse_tracker.reset()
        self._last_mouse_update = self._blank_mouse_update()
        self._mouse_mode_enabled = False
        self._mouse_control_text = "Mouse mode off" if self.mouse_controller.available else self.mouse_controller.message
        self._mouse_status_text = "off" if self.mouse_controller.available else "unavailable"
        self._reset_chrome_wheel(clear_cooldown=True)
        self._reset_spotify_wheel(clear_cooldown=True)
        self._reset_youtube_wheel(clear_cooldown=True)
        self.chrome_router.reset()
        self._chrome_mode_enabled = False
        self._chrome_control_text = self.chrome_controller.message
        self._last_chrome_action = "-"
        self.spotify_router.reset()
        self._spotify_control_text = self.spotify_controller.message
        self._spotify_info_text = "-"
        self._last_spotify_action = "-"
        self.youtube_router.reset()
        self._youtube_control_text = "YouTube idle"
        self._youtube_mode_info = "off"
        self._last_chrome_action_counter = 0
        self._last_spotify_action_counter = 0
        self._last_youtube_action_counter = 0
        self._reset_voice_state()
        self._reset_drawing_runtime()
        self._reset_drawing_wheel(clear_cooldown=True)
        self._reset_utility_wheel(clear_cooldown=True)
        self._camera_draw_canvas = None
        self._camera_draw_history = []
        self._camera_draw_strokes = []
        self._camera_draw_active_stroke_points = []
        self._camera_draw_raster_dirty = False
        self._camera_draw_last_point = None
        self._camera_draw_erasing = False
        self._drawing_request_action = ""
        self._drawing_swipe_cooldown_until = 0.0
        self._drawing_request_token = 0
        self._utility_request_action = ""
        self._utility_request_token = 0
        self._utility_recording_active = False
        self._utility_recording_stop_candidate_since = 0.0
        self._utility_capture_selection_active = False
        self._utility_capture_cursor_norm = None
        self._utility_capture_left_down = False
        self._utility_capture_right_down = False
        self._utility_capture_clicks_armed = False
        self._reset_window_gesture_state(clear_cooldown=True)

        camera_info, cap = self._open_camera()
        if cap is None or camera_info is None:
            # v1.1.7 UX rewrite: never blame specific apps in the
            # user-visible error. The camera-lock detector still
            # runs — but only to stderr for our diagnostics. A user
            # who just installed Razer Synapse (per our own Kiyo Pro
            # fps guidance) shouldn't be told to close it. The status
            # and error emit stay neutral so the Stop/Start retry
            # entrypoint is the only action they need.
            try:
                from ..camera.camera_lock_detector import (
                    detect_camera_holding_processes,
                    summarize_processes_for_user,
                )
                detected_holders = detect_camera_holding_processes()
                if detected_holders:
                    holder_summary = summarize_processes_for_user(detected_holders)
                    sys.stderr.write(
                        f"[camera-diag] engine open failed; camera-sharing apps "
                        f"currently running: {holder_summary}. Not surfacing to user.\n"
                    )
                    sys.stderr.flush()
            except Exception:
                pass
            self._emit_status("no camera found")
            self.error_occurred.emit(
                "Touchless couldn't reach a camera. Make sure your webcam is "
                "plugged in, then press Start again."
            )
            self.running_state_changed.emit(False)
            return

        self._camera_info = camera_info
        self._cap = cap
        self._running = True
        self._last_time = time.time()
        self._fps = 0.0
        self._low_fps_last_process = 0.0
        self._fullscreen_foreground_active = False
        self._fullscreen_foreground_process = ""
        self._fullscreen_check_last = 0.0
        self._dynamic_hold_label = "neutral"
        self._dynamic_hold_until = 0.0
        # Camera-health tracking. _camera_first_frame_received flips
        # True on the first successful cap.read() of this session and
        # gates the "Starting camera…" → "Touchless active" status
        # update. _camera_recovery_* fields drive the dead-cap
        # auto-reopen path in _tick: if isOpened() goes False after
        # frames were flowing, the engine tries to reopen the same
        # camera index (up to MAX) before giving up. Covers transient
        # USB drops, brief driver hiccups, and the "another app
        # grabbed the camera for a second" case.
        self._camera_first_frame_received = False
        self._camera_start_at = time.monotonic()
        self._camera_recovery_attempts = 0
        self._camera_recovery_last_at = 0.0
        self._camera_recovery_max_attempts = 3
        self._camera_recovery_cooldown_s = 1.5
        self._camera_no_first_frame_warned = False
        self.camera_selected.emit(camera_info.display_name)
        self._emit_status("starting camera…")
        self.command_detected.emit("Starting camera…")
        self.running_state_changed.emit(True)
        self._timer.start()

        def _warm_spotify() -> None:
            try:
                self.spotify_controller.warm_up()
            except Exception:
                pass

        import threading
        threading.Thread(target=_warm_spotify, daemon=True).start()

    def stop(self) -> None:
        if not self._running and self._cap is None and self.engine is None:
            return
        self._shutdown_runtime(emit_signal=True)

    def set_tutorial_block_mute(self, blocked: bool) -> None:
        """Toggle whether the engine should drop the shaka-mute
        trigger. Used by the tutorial's volume step to keep the OS
        audio from actually muting before the user has practised
        up + down direction; tutorial flips the flag back off once
        both swings are complete."""
        self._tutorial_block_mute = bool(blocked)

    def set_pipeline_frozen(self, frozen: bool) -> None:
        """Freeze / unfreeze the gesture pipeline. While frozen,
        _tick still emits raw_frame_ready (so the custom-gesture
        recorder keeps getting frames over its existing connection)
        but skips MediaPipe inference, gesture-action dispatch, and
        the debug-payload emit. Live-view widgets observe
        frozen_state_changed to render their paused/blurred overlay.
        Idempotent â€” repeated set_pipeline_frozen(True) calls don't
        re-emit, so the receiver doesn't churn its overlay state."""
        new_state = bool(frozen)
        if new_state == self._frozen:
            return
        self._frozen = new_state
        try:
            self.frozen_state_changed.emit(new_state)
        except Exception:
            pass

    def _shutdown_runtime(self, *, emit_signal: bool) -> None:
        self._timer.stop()
        # Stop the background engine runner BEFORE closing the engine,
        # otherwise an in-flight process_frame call could touch a
        # closed MediaPipe / ONNX session and crash. The runner's stop
        # joins its thread (up to 2 s), guaranteeing no further calls
        # into the engine after this returns.
        try:
            self._engine_runner.stop()
        except Exception:
            pass
        # Drop the engine reference inside the runner so it doesn't
        # keep the soon-to-be-closed engine alive.
        self._engine_runner.set_engine(None)
        # Reset async-result bookkeeping. A queued signal still in
        # flight will be a no-op once it lands (because _running flips
        # to False below), but clearing the flag here means a fresh
        # start() begins from a known-clean state.
        self._async_result_pending = False
        self._tick_timing_state = None
        if self._cap is not None:
            # Don't release the phone-camera-QR capture â€” it's owned by the
            # MainWindow/PhoneCameraServer and must survive engine restarts
            # (e.g. toggling Low FPS Mode re-opens the camera via start()).
            if self._cap is not self._phone_camera_capture:
                self._cap.release()
            self._cap = None
        # C17: drain the engine cache before nulling self.engine — some
        # of the entries may be the currently active engine, others are
        # inactive cached siblings. Closing all of them here matches the
        # pre-cache behavior where every mode swap closed its old engine
        # immediately. Never leak an engine across engine-restarts.
        if self._engine_cache is not None:
            for cached_engine in self._engine_cache.values():
                try:
                    cached_engine.close()
                except Exception:
                    pass
            self._engine_cache = None
        if self.engine is not None:
            try:
                # If self.engine was already closed via the cache drain
                # above, close() is idempotent — the underlying MediaPipe
                # / ONNX close() call short-circuits on already-closed
                # state. Kept for the (rare) case where the engine was
                # never added to the cache.
                self.engine.close()
            except Exception:
                pass
            self.engine = None
        self._camera_info = None
        self._running = False
        self._fps = 0.0
        self._low_fps_last_process = 0.0
        self._dynamic_hold_label = "neutral"
        self._dynamic_hold_until = 0.0
        self._volume_mode_active = False
        self._volume_overlay_visible = False
        self.volume_overlay.hide_overlay()
        self.voice_status_overlay.hide_overlay()
        try:
            self.low_fps_suggestion_overlay.hide()
        except Exception:
            pass
        self._low_fps_suggest_visible = False
        self._low_fps_suggest_below_since = None
        self.mouse_controller.release_all()
        self.mouse_tracker.reset()
        self._last_mouse_update = self._blank_mouse_update()
        self._mouse_mode_enabled = False
        self._mouse_control_text = "Mouse mode off" if self.mouse_controller.available else self.mouse_controller.message
        self._mouse_status_text = "off" if self.mouse_controller.available else "unavailable"
        self._reset_chrome_wheel(clear_cooldown=True)
        self._reset_spotify_wheel(clear_cooldown=True)
        self._reset_youtube_wheel(clear_cooldown=True)
        self.chrome_router.reset()
        self.spotify_router.reset()
        self.volume_tracker.reset(self._volume_level, self._volume_muted)
        self._reset_voice_state()
        self._reset_drawing_runtime()
        self._reset_drawing_wheel(clear_cooldown=True)
        self._reset_utility_wheel(clear_cooldown=True)
        self._camera_draw_canvas = None
        self._camera_draw_history = []
        self._camera_draw_strokes = []
        self._camera_draw_active_stroke_points = []
        self._camera_draw_raster_dirty = False
        self._camera_draw_last_point = None
        self._camera_draw_erasing = False
        self._drawing_request_action = ""
        self._drawing_swipe_cooldown_until = 0.0
        self._utility_request_action = ""
        self._utility_recording_active = False
        self._utility_recording_stop_candidate_since = 0.0
        self._utility_capture_selection_active = False
        self._utility_capture_cursor_norm = None
        self._utility_capture_left_down = False
        self._utility_capture_right_down = False
        self._utility_capture_clicks_armed = False
        self._reset_window_gesture_state(clear_cooldown=True)
        self._emit_status("idle")
        if emit_signal:
            self.running_state_changed.emit(False)

    def set_phone_camera_capture(self, capture) -> None:
        """Hand the engine a running PhoneCameraCapture from the QR-dialog flow.

        When set, _open_camera() will use this capture in place of any
        local-device or URL-based source. Callers set None to clear it
        (e.g. when the user disconnects the phone).
        """
        self._phone_camera_capture = capture

    def _open_camera(self):
        # QR-dialog path wins when present: the server is already running
        # on a worker thread and frames are flowing; we just wrap its
        # capture in the (info, cap) shape the engine expects. A local
        # camera_index_override still beats this (used by the in-app
        # camera test flow).
        phone_qr_capture = getattr(self, "_phone_camera_capture", None)
        phone_url = str(getattr(self.config, "phone_camera_url", "") or "").strip()
        use_phone_url = bool(getattr(self.config, "phone_camera_enabled", False)) and phone_url
        if self.camera_index_override is not None:
            result = open_camera_by_index(
                self.camera_index_override,
                max_index=self.config.camera_scan_limit,
                **self._darwin_open_kwargs(),
            )
            # Fallback: if the requested index doesn't open (the device
            # disappeared, the OpenCV backend that worked at scan time
            # rejects it now, etc.) fall back to whatever the preferred
            # / first-available helper finds. Without this, a user
            # whose saved camera fails to open hits a black "no camera"
            # screen even though other cameras are present â€” exactly
            # the start-button-does-nothing report from the field.
            if result is None or result[1] is None:
                result = open_preferred_or_first_available(
                    self.config.preferred_camera_index,
                    max_index=self.config.camera_scan_limit,
                    **self._darwin_open_kwargs(),
                )
        elif phone_qr_capture is not None and phone_qr_capture.isOpened():
            info = SimpleNamespace(
                index=-2,
                backend=0,
                backend_name="PhoneQR",
                display_name="Phone Camera (QR)",
            )
            result = (info, phone_qr_capture)
        elif use_phone_url:
            result = open_phone_camera_url(phone_url)
            if result[1] is None:
                # Phone camera unreachable at startup â€” fall back to the last
                # preferred local camera so the app can still run. The UI will
                # reflect this via the live-status path (status_changed signal).
                result = open_preferred_or_first_available(
                    self.config.preferred_camera_index,
                    max_index=self.config.camera_scan_limit,
                    **self._darwin_open_kwargs(),
                )
        else:
            result = open_preferred_or_first_available(
                self.config.preferred_camera_index,
                max_index=self.config.camera_scan_limit,
                **self._darwin_open_kwargs(),
            )
        self._apply_low_fps_capture_tuning(result)
        # C20: default-mode tuning — nudge the OpenCV cap to 30 fps +
        # MJPG. No-op for Low FPS (already tuned above) and for Lite /
        # GPU (they upgrade to ffmpeg-MJPG right after this).
        self._apply_default_capture_tuning(result)
        result = self._upgrade_to_ffmpeg_capture_if_lite(result)
        return result

    def _attempt_camera_recovery(self) -> None:
        """Reopen the camera after the threaded reader flagged the
        capture dead. Rate-limited so we don't spin (the open call
        itself can take 500 ms+ on a slow virtual camera) and capped
        at _camera_recovery_max_attempts so a permanently-gone camera
        eventually surfaces as a hard error instead of looping.

        Same backend/source as the current cap — uses
        camera_index_override if it was set, otherwise the same
        preferred-or-first-available path the original open went
        through. The threaded reader's warmup discard runs again on
        the new cap, so the consumer still doesn't see partial frames
        after a recovery.
        """
        now = time.monotonic()
        if now - self._camera_recovery_last_at < self._camera_recovery_cooldown_s:
            return
        self._camera_recovery_last_at = now
        # Detect known camera-holding apps every recovery attempt so
        # the user's status pill shows the specific culprit ("Razer
        # Synapse 3 is holding your camera") instead of a generic
        # "no camera" message. When the user closes that app, the
        # next recovery cycle succeeds automatically — zero clicks
        # inside Touchless. Detection is cheap (~150 ms one-shot to
        # tasklist) and only runs when recovery is actually needed,
        # so no impact on the happy path.
        # v1.1.7 UX rewrite: run the detector for stderr diagnostics
        # but keep user-facing text neutral. Never blame a specific
        # app in the recovery status.
        detected_holders: list = []
        try:
            from ..camera.camera_lock_detector import (
                detect_camera_holding_processes,
                summarize_processes_for_user,
            )
            detected_holders = detect_camera_holding_processes()
            if detected_holders:
                holder_summary = summarize_processes_for_user(detected_holders)
                sys.stderr.write(
                    f"[camera-diag] recovery in progress; camera-sharing apps "
                    f"currently running: {holder_summary}. Not surfacing to user.\n"
                )
                sys.stderr.flush()
        except Exception:
            detected_holders = []
        if self._camera_recovery_attempts >= self._camera_recovery_max_attempts:
            # Give up — clean shutdown rather than spinning forever.
            if self._camera_recovery_attempts == self._camera_recovery_max_attempts:
                # Increment once more so this branch only emits once.
                self._camera_recovery_attempts += 1
                try:
                    self._emit_status("camera disconnected")
                    self.error_occurred.emit(
                        "Camera connection lost. Press Stop and Start to try again."
                    )
                except Exception:
                    pass
            return
        self._camera_recovery_attempts += 1
        try:
            self._emit_status(
                f"reconnecting camera (attempt {self._camera_recovery_attempts}/"
                f"{self._camera_recovery_max_attempts})…"
            )
        except Exception:
            pass
        # Tear down the dead cap before opening a new one — some
        # drivers refuse a second open while a stale handle is still
        # holding the device.
        try:
            if self._cap is not None:
                self._cap.release()
        except Exception:
            pass
        self._cap = None
        try:
            camera_info, cap = self._open_camera()
        except Exception:
            camera_info, cap = (None, None)
        if cap is None or camera_info is None:
            # Open failed — leave _cap None and try again on the next
            # tick (after the cooldown). _camera_recovery_attempts
            # already incremented above so we'll hit the max-attempts
            # branch eventually if the camera doesn't come back.
            return
        self._camera_info = camera_info
        self._cap = cap
        # Reset first-frame tracking so the recovered cap goes
        # through the same "starting → active" status transition the
        # initial open did. Don't reset attempts — that only resets
        # after a successful frame arrives (in _tick).
        self._camera_first_frame_received = False
        self._camera_start_at = now
        self._camera_no_first_frame_warned = False
        try:
            self.camera_selected.emit(camera_info.display_name)
        except Exception:
            pass

    def _upgrade_to_ffmpeg_capture_if_lite(self, open_result):
        # When Lite Mode is on for a local USB webcam, swap the
        # OpenCV cap for an ffmpeg-backed MJPG cap. ffmpeg's dshow
        # input reliably forces MJPG where OpenCV silently keeps
        # YUY2, which is the only path on Windows that breaks the
        # ~30 fps ceiling at 720p (and the ~10 fps ceiling at 1080p)
        # for cheap webcams with USB-bandwidth-bound raw streams.
        # If the ffmpeg cap can't deliver a frame within its startup
        # budget we keep the original OpenCV cap, so the user is
        # never stranded without video. Diagnostic prints to stderr
        # so the user / dev can confirm in one shot which path was
        # actually engaged when troubleshooting low-fps reports.
        def _log(msg: str) -> None:
            try:
                sys.stderr.write(f"[lite_mode/ffmpeg] {msg}\n")
                sys.stderr.flush()
            except Exception:
                pass

        # Camera-side ffmpeg/MJPG swap fires for Lite Mode, GPU Mode,
        # OR Low FPS Mode (manual or auto-engaged). Without this, a
        # user on slow hardware who hits the auto-engage path
        # (CRITICAL_FPS_THRESHOLD < 12 fps) gets the MediaPipe
        # complexity drop but keeps OpenCV's YUY2 30 fps camera
        # ceiling — and at the YUY2 bandwidth cap, the camera often
        # only delivers 8-10 fps at 720p, which then becomes the
        # actual frame ceiling regardless of inference speedup.
        # Adding low_fps to the gate is what makes auto-engage on
        # dad's-PC-class hardware (GTX 960, no GPU mode set, fps
        # auto-degraded to low_fps) actually translate the inference
        # speedup into more live FPS. Previously the ffmpeg upgrade
        # was EXPLICITLY skipped when low_fps was active — that
        # branch is now removed.
        if not (self._perf_optimisations_enabled() or self._low_fps_active):
            _log("skipped: lite_mode + gpu_mode + low_fps_mode all off")
            return open_result
        if not isinstance(open_result, tuple) or len(open_result) < 2:
            _log("skipped: open_result not in (info, cap) shape")
            return open_result
        info, cap = open_result[0], open_result[1]
        if cap is None or info is None:
            _log("skipped: cap or info is None")
            return open_result
        index = getattr(info, "index", None)
        if index is None or int(index) < 0:
            _log(f"skipped: index={index!r} (phone-camera path)")
            return open_result
        if not sys.platform.startswith("win"):
            _log("skipped: platform is not windows")
            return open_result
        display_name = str(getattr(info, "display_name", "") or "")
        # Verbose: dump the full ffmpeg dshow device list so the
        # user can see what was offered when "device='USB Video
        # Device'" looks suspicious for a Razer / Logitech webcam.
        try:
            from ..camera.ffmpeg_capture import list_dshow_video_devices

            _log(
                f"qt_hint={display_name!r} index={index} "
                f"ffmpeg_devices={list_dshow_video_devices()!r}"
            )
        except Exception:
            pass
        device_name = resolve_dshow_device_for_index(int(index), qt_name_hint=display_name)
        if not device_name:
            _log(
                f"skipped: could not resolve dshow device for index={index} "
                f"qt_hint={display_name!r}"
            )
            return open_result
        _log(f"opening ffmpeg cap: device={device_name!r} 1280x720 @ 60 fps MJPG")
        # Release the OpenCV cap BEFORE starting ffmpeg. Windows
        # DirectShow gives exclusive capture access to one process at
        # a time on most consumer webcams â€” if OpenCV still holds the
        # device, ffmpeg's open will fail with "device busy" and we
        # silently fall through to OpenCV's slow path. So we let go
        # of the camera first, attempt ffmpeg, and if ffmpeg can't
        # start we re-open OpenCV from scratch so the user keeps
        # video.
        index_int = int(index)
        try:
            cap.release()
        except Exception:
            pass
        # C27: match _apply_perf_camera_path — 640x480 not 1280x720.
        # Same rationale: kill the persistent 1-2 s visual lag the
        # user reports on GPU Mode by keeping frame size at parity
        # with Default's OpenCV cap.
        # r54 v9: pre-flight short shutter (see helper docstring).
        # v1.1.7.1: luma_min_threshold=50 downshifts 60→30 fps when
        # the driver produces a too-dark frame at the higher rate
        # (Subodh Tope's HP HD Camera 2026-08-16 report). Safety net
        # only — bright frames pass first try, dim ones fall back.
        self._preflight_short_shutter_for_ffmpeg(index_int, device_name)
        ffmpeg_cap = open_ffmpeg_cap_with_fps_fallback(
            device_name, width=640, height=480,
            luma_min_threshold=50.0,
        )
        if ffmpeg_cap is not None and ffmpeg_cap.isOpened():
            _log("ffmpeg cap engaged")
            return (info, ffmpeg_cap)
        _log("ffmpeg cap startup failed â€” falling back to a fresh OpenCV cap")
        if ffmpeg_cap is not None:
            try:
                ffmpeg_cap.release()
            except Exception:
                pass
        # v1.1.7.2 (dad rig): release the preflight EXPOSURE latch
        # before opening the OpenCV fallback cap. If we don't, the
        # driver keeps EXPOSURE=-6 across the cap re-open and the
        # OpenCV cap opens dark, exactly like dad's 2026-08-19 log
        # showed. The release helper writes EXP=-4 then AUTO=auto
        # so the driver leaves manual mode with a bright exposure.
        try:
            self._release_ffmpeg_preflight(index_int)
        except Exception:
            pass
        # DirectShow-release race guard: the OpenCV cap was released
        # just moments ago (line above the ffmpeg attempt) so the
        # Windows DSHOW filter may still be tearing down. Reopening
        # too quickly returns a half-dead cap that reports isOpened()
        # = False. With the old 8 s ffmpeg timeout this was never
        # observed because by the time ffmpeg gave up the driver had
        # long since freed the device. The new 2.5 s timeout opens
        # the race on slow-release webcams (Razer Kiyo Pro is the
        # one we observed in user logs â€” it can take 1.5-2 s to
        # finish releasing). Retry up to 8 times with 250 ms gaps:
        # total worst-case added latency 2 s, which is still half
        # of the 4 s saved by dropping the per-attempt ffmpeg
        # timeout from 8 s to 2.5 s, AND converges on the first
        # retry on cameras that release fast.
        recovered = None
        for retry_idx in range(8):
            recovered = open_camera_by_index(index_int, max_index=self.config.camera_scan_limit)
            if (
                isinstance(recovered, tuple)
                and len(recovered) >= 2
                and recovered[1] is not None
            ):
                if retry_idx > 0:
                    _log(f"OpenCV reopen succeeded on retry {retry_idx}")
                # v1.1.7.2 (dad rig): re-run the standard tuning on
                # the fresh OpenCV cap so should_kick / luma verify
                # get a chance to correct any lingering manual state.
                try:
                    self._apply_default_capture_tuning(recovered)
                except Exception:
                    pass
                return recovered
            time.sleep(0.25)
        # Last-ditch: open with Media Foundation so we have *some*
        # camera object â€” even default-format YUY2 is better than
        # leaving the engine with self._cap = None. Explicit MSMF
        # (rather than CAP_ANY whose Windows resolution varies by
        # OpenCV build) keeps this path consistent with the
        # MSMF-first policy in camera_utils._backend_candidates(),
        # so a buggy DirectShow filter can't sneak back in via the
        # default-backend resolution.
        try:
            backend = getattr(cv2, "CAP_MSMF", getattr(cv2, "CAP_ANY", 0))
            recovered_cap = cv2.VideoCapture(index_int, backend)
            if recovered_cap.isOpened():
                _log("MSMF last-ditch reopen engaged")
                return (info, recovered_cap)
        except Exception:
            pass
        _log(
            f"all OpenCV reopens failed after ffmpeg startup miss â€” "
            f"engine will report no-camera and tutorial/UI will show 'runtime stopped'"
        )
        return open_result

    def _draw_low_fps_badge(self, frame) -> None:
        try:
            height, width = frame.shape[:2]
        except Exception:
            return
        text = "Low FPS Mode"
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.6
        thickness = 2
        (text_w, text_h), baseline = cv2.getTextSize(text, font, scale, thickness)
        pad_x, pad_y = 12, 8
        x1 = 14
        y1 = 14
        x2 = x1 + text_w + pad_x * 2
        y2 = y1 + text_h + pad_y * 2
        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 215, 255), thickness=-1)
        cv2.addWeighted(overlay, 0.82, frame, 0.18, 0, frame)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 120, 170), thickness=2)
        cv2.putText(
            frame,
            text,
            (x1 + pad_x, y2 - pad_y - 2),
            font,
            scale,
            (20, 20, 20),
            thickness,
            cv2.LINE_AA,
        )

    def _apply_low_fps_capture_tuning(self, open_result) -> None:
        if not self._low_fps_active:
            return
        cap = None
        if isinstance(open_result, tuple) and len(open_result) >= 2:
            cap = open_result[1]
        elif open_result is not None and hasattr(open_result, "set"):
            cap = open_result
        if cap is None:
            return
        try:
            if hasattr(cv2, "CAP_PROP_BUFFERSIZE"):
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
        try:
            if hasattr(cv2, "CAP_PROP_FOURCC"):
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        except Exception:
            pass
        try:
            if hasattr(cv2, "CAP_PROP_FPS"):
                cap.set(cv2.CAP_PROP_FPS, 30)
        except Exception:
            pass
        try:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        except Exception:
            pass

    def _apply_macos_short_shutter(self, cap) -> tuple:
        """AVFoundation short-shutter for the Camera Boost checkbox.

        Windows writes DSHOW log2 EV (AUTO=1.0, EXPOSURE=-6). Those
        units are meaningless on AVFoundation, so the checkbox did
        nothing on Mac. Here:
          1. OpenCV: AUTO_EXPOSURE=0 (manual) + EXPOSURE≈1/60s.
          2. Device-level AVCaptureDevice custom exposure (FaceTime
             often ignores OpenCV property writes).
        Best-effort; logs the outcome. Returns (ok_auto, ok_exp).
        """
        ok_auto = None
        ok_exp = None
        try:
            if hasattr(cv2, "CAP_PROP_AUTO_EXPOSURE"):
                ok_auto = cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0)
        except Exception as exc:
            try:
                sys.stderr.write(
                    f"[r49-short-shutter] mac AUTO write: {type(exc).__name__}: {exc}\n"
                )
                sys.stderr.flush()
            except Exception:
                pass
        try:
            if hasattr(cv2, "CAP_PROP_EXPOSURE"):
                ok_exp = cap.set(cv2.CAP_PROP_EXPOSURE, 0.016)
        except Exception as exc:
            try:
                sys.stderr.write(
                    f"[r49-short-shutter] mac EXP write: {type(exc).__name__}: {exc}\n"
                )
                sys.stderr.flush()
            except Exception:
                pass
        try:
            from AVFoundation import (  # type: ignore
                AVCaptureDevice,
                AVCaptureExposureModeCustom,
                AVCaptureFocusModeLocked,
            )
            device = None
            try:
                devices = list(AVCaptureDevice.devicesWithMediaType_("vide") or [])
            except Exception:
                devices = []
            if devices:
                device = devices[0]
            if device is None:
                device = AVCaptureDevice.defaultDeviceWithMediaType_("vide")
            if device is not None:
                locked = device.lockForConfiguration_(None)
                if isinstance(locked, tuple):
                    locked = locked[0]
                if locked:
                    try:
                        min_dur = device.activeFormat().minExposureDuration()
                        from CoreMedia import CMTimeMake, CMTimeCompare  # type: ignore

                        want = CMTimeMake(1, 90)
                        duration = want
                        try:
                            if CMTimeCompare(want, min_dur) < 0:
                                duration = min_dur
                        except Exception:
                            duration = want
                        iso = float(device.ISO())
                        min_iso = float(device.activeFormat().minISO())
                        max_iso = float(device.activeFormat().maxISO())
                        iso = max(min_iso, min(max_iso, iso * 1.25))
                        try:
                            if device.isFocusModeSupported_(AVCaptureFocusModeLocked):
                                device.setFocusMode_(AVCaptureFocusModeLocked)
                        except Exception:
                            pass
                        if not device.isExposureModeSupported_(AVCaptureExposureModeCustom):
                            sys.stderr.write(
                                "[r49-short-shutter] mac AVCaptureDevice: "
                                "custom exposure not supported on "
                                f"{device.localizedName()!r}; "
                                "640x480 reopen is the fps lever\n"
                            )
                            sys.stderr.flush()
                        else:
                            device.setExposureMode_(AVCaptureExposureModeCustom)
                            device.setExposureModeCustomWithDuration_ISO_completionHandler_(
                                duration, iso, None
                            )
                            sys.stderr.write(
                                "[r49-short-shutter] mac AVCaptureDevice custom "
                                f"exposure 1/90 ISO={iso:.0f} af=locked "
                                f"device={device.localizedName()!r}\n"
                            )
                            sys.stderr.flush()
                    finally:
                        device.unlockForConfiguration()
        except Exception as exc:
            try:
                sys.stderr.write(
                    f"[r49-short-shutter] mac AVCaptureDevice: "
                    f"{type(exc).__name__}: {exc}\n"
                )
                sys.stderr.flush()
            except Exception:
                pass
        return ok_auto, ok_exp

    def _apply_default_capture_tuning(self, open_result) -> None:
        """Push a 30 fps + MJPG FOURCC hint into the OpenCV cap on
        initial open when we're in default mode (no perf toggle).

        Motivation (v1.1.7 C20): the user reports steady 20 fps in
        default mode on a Kiyo Pro that the DShow probe confirms can
        do 60 fps at every 640×480 / 720p / 1080p mode. cv2.VideoCapture
        never sends the DShow driver an explicit framerate request on
        open — it takes whatever the driver defaults to. The default
        for many UVC drivers under YUY2 is 15-20 fps because that's
        the exposure-time-safe rate under Windows auto-exposure. An
        explicit CAP_PROP_FPS=30 request lifts that cap on drivers
        that honour it (including the Kiyo Pro's).

        MJPG FOURCC hint is best-effort: OpenCV's Windows path often
        silently ignores it, but when it works the driver switches
        to MJPG which lifts the YUY2 bandwidth cap. Zero cost when
        ignored.

        v1.1.7 C25: also applies to Lite Mode (which now shares the
        default OpenCV cap since C24) with an fps request bumped to
        60 fps instead of 30. On drivers that honour it (Kiyo Pro
        does), Lite delivers noticeably higher fps than Default at
        the cost of higher camera-decode CPU per second — matches
        the user's "Lite is a universal CPU-cost fps boost" intent.
        On drivers that ignore CAP_PROP_FPS (Lite is no worse than
        Default in that case), it's a silent no-op.

        Skipped when GPU Mode is on (ffmpeg-MJPG handles fps itself)
        or Low FPS is active (its own tuning path targets 30 fps to
        keep CPU low on weak hardware).
        """
        if self._wants_ffmpeg_cap() or self._low_fps_active:
            # r49 diag: log why the tuning was skipped so a support
            # reader can tell 'toggle was dead' from 'toggle never
            # fired'. Without this line, GPU-Mode / Low-FPS sessions
            # produce ZERO r49 breadcrumbs and look identical to a
            # session where the code path never executed.
            try:
                _reason = "wants_ffmpeg" if self._wants_ffmpeg_cap() else "low_fps_active"
                sys.stderr.write(
                    f"[r49-short-shutter] skip: reason={_reason} "
                    f"(exposure hint is dead in this mode)\n"
                )
                sys.stderr.flush()
            except Exception:
                pass
            return
        cap = None
        if isinstance(open_result, tuple) and len(open_result) >= 2:
            cap = open_result[1]
        elif open_result is not None and hasattr(open_result, "set"):
            cap = open_result
        if cap is None:
            return
        # Skip if the cap is our ffmpeg wrapper (already 60 fps MJPG).
        try:
            if "FfmpegMjpegCapture" in type(cap).__name__:
                try:
                    sys.stderr.write(
                        "[r49-short-shutter] skip: reason=ffmpeg_cap "
                        "(exposure hint is dead on ffmpeg cap)\n"
                    )
                    sys.stderr.flush()
                except Exception:
                    pass
                return
        except Exception:
            pass
        # MJPG FOURCC hint. Best-effort — many Windows OpenCV builds
        # silently keep YUY2. When it lands, the driver can hit higher
        # fps at the same resolution than YUY2 bandwidth would allow.
        # Darwin: skip — AVFoundation FaceTime cameras do not speak
        # MJPG FOURCC the way DSHOW does; the write is a no-op or
        # can knock the session into a worse mode.
        try:
            if sys.platform != "darwin" and hasattr(cv2, "CAP_PROP_FOURCC"):
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        except Exception:
            pass
        # Explicit fps request. UVC drivers honour this when the
        # requested rate is within their advertised range for the
        # current (fmt, resolution) combo. C25: Lite gets 60 fps
        # request (matches the mode's "fps boost" intent); Default
        # keeps 30 fps. Darwin: do NOT write FPS or size on a live
        # session — AVFoundation renegotiates into a ~11 fps mode.
        # Size is requested in try_open_camera before the first read.
        try:
            if sys.platform != "darwin" and hasattr(cv2, "CAP_PROP_FPS"):
                lite_active = bool(getattr(self.config, "lite_mode", False))
                fps_target = 60.0 if lite_active else 30.0
                cap.set(cv2.CAP_PROP_FPS, fps_target)
        except Exception:
            pass
        # Small buffer size so a brief stall doesn't accumulate
        # latency in the driver's queue.
        try:
            if hasattr(cv2, "CAP_PROP_BUFFERSIZE"):
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
        # r49: opt-in short-shutter hint (was C31 unconditional,
        # r47 reverted to no-op). We now gate it on the config flag
        # camera_force_short_shutter (default False) so premium
        # cameras like the Razer Kiyo Pro do NOT trip the 60->25 fps
        # HDR/auto-exposure throttle by default, while dad's cheap
        # Realtek UVC can flip it ON in Settings -> General to stop
        # its driver from holding the shutter open for 40-80 ms per
        # frame in indoor light. Env HGR_FORCE_SHORT_SHUTTER=0/1
        # overrides the config for A/B without touching settings.
        #
        # Value semantics WARNING (documented for the log reader):
        # OpenCV's DSHOW backend historically uses 0.25=manual /
        # 0.75=auto for CAP_PROP_AUTO_EXPOSURE. MSMF and V4L2 use
        # 1/3. We write 1.0 here; on DSHOW the driver may clamp to
        # 0.75 (which is AUTO -- opposite of intent). The post-hint
        # readback below logs cap.getBackendName() so a support
        # reader can interpret the readback correctly and know
        # whether to A/B with HGR_FORCE_SHORT_SHUTTER=0 vs =1.
        # r50: pull display_name for the classifier. At the initial
        # cap-open call site self._camera_info is still None (assigned
        # AFTER _open_camera returns). Prefer the fresh info from the
        # open_result tuple, fall back to the live self._camera_info
        # for the three mid-session re-tune callers, guard with getattr
        # so no path can AttributeError.
        _display_name = ""
        try:
            fresh_info = None
            if isinstance(open_result, tuple) and len(open_result) >= 1:
                fresh_info = open_result[0]
            _display_name = getattr(fresh_info, "display_name", None) or getattr(
                self._camera_info, "display_name", None
            ) or ""
        except Exception:
            _display_name = ""
        # r50: classifier verdict. None = unknown, True = generic UVC
        # (auto-apply hint), False = premium (skip hint).
        _classifier_verdict = None
        try:
            from ..camera.camera_utils import classify_camera_shutter_hint as _clf
            _classifier_verdict = _clf(_display_name)
        except Exception:
            _classifier_verdict = None
        # v1.1.7.1 Layer 3 sticky rollback: if we already auto-applied
        # the hint on this camera THIS SESSION and rolled it back on
        # darkness verification, suppress the classifier verdict so we
        # don't re-apply and re-darken on every settings toggle / mode
        # re-tune. User can still force it via the explicit config
        # checkbox — this suppression only demotes the classifier's
        # auto-verdict, not the user's explicit choice.
        try:
            _rb_set = getattr(self, "_r49_auto_rollback_cameras", None)
            if _rb_set:
                _display_key_gate = str(_display_name or "").lower().strip()
                if _display_key_gate and _display_key_gate in _rb_set:
                    if _classifier_verdict is True:
                        _classifier_verdict = None
                    try:
                        sys.stderr.write(
                            f"[r49-short-shutter] classifier verdict SUPPRESSED "
                            f"(prior auto-rollback this session) "
                            f"camera={_display_key_gate!r}\n"
                        )
                        sys.stderr.flush()
                    except Exception:
                        pass
        except Exception:
            pass
        env_override = None
        _decision_reason = "default_off"
        try:
            import os as _os
            env_override = _os.environ.get("HGR_FORCE_SHORT_SHUTTER")
            _user_chose = bool(
                getattr(self.config, "camera_force_short_shutter_user_chose", False)
            )
            _cfg_ss = bool(getattr(self.config, "camera_force_short_shutter", False))
            if env_override is not None:
                apply_hint = env_override.strip() not in ("", "0", "false", "False")
                _decision_reason = "env"
            elif _user_chose:
                # r53: user has explicitly toggled the checkbox — respect
                # whatever they set, don't let the classifier override.
                apply_hint = _cfg_ss
                _decision_reason = "config_user_chose" if _cfg_ss else "config_user_chose_off"
            elif _cfg_ss:
                apply_hint = True
                _decision_reason = "config"
            elif _classifier_verdict is True:
                apply_hint = True
                _decision_reason = "classifier_generic"
            elif _classifier_verdict is False:
                apply_hint = False
                _decision_reason = "classifier_premium"
            else:
                apply_hint = False
                _decision_reason = "default_off"
        except Exception:
            apply_hint = False
            _decision_reason = "default_off"
        # Stash for the r50 post_hint_fps breadcrumb + first-fire pill.
        self._r50_last_display_name = _display_name
        self._r50_last_decision_reason = _decision_reason
        self._r50_last_classifier_verdict = _classifier_verdict
        # r49 diag Q1 (extended in r50): gate evaluation -- always
        # emitted so the log answers 'did the gate open?' independent
        # of what happens after. Extended in r50 with display_name +
        # classifier verdict + decision reason so Kiyo Pro's
        # 'deliberately did nothing' is captured in the log as
        # non-regression proof.
        try:
            _cfg_ss = bool(getattr(self.config, "camera_force_short_shutter", False))
            sys.stderr.write(
                f"[r49-short-shutter] evaluate: env={env_override!r} "
                f"config.camera_force_short_shutter={_cfg_ss} "
                f"display_name={_display_name!r} "
                f"classifier={_classifier_verdict} "
                f"reason={_decision_reason} "
                f"-> apply_hint={apply_hint}\n"
            )
            sys.stderr.flush()
        except Exception:
            pass
        if apply_hint:
            # r49 diag Q2: pre/post driver readback + backend name.
            # cap.set() returns True on many UVC drivers even when
            # the value is silently rejected; only cap.get() after
            # tells the truth. Backend name disambiguates DSHOW's
            # 0.25/0.75 magic values from MSMF/V4L2's 1/3.
            _backend_name = None
            try:
                _backend_name = cap.getBackendName()
            except Exception:
                pass
            # r51: capture the pre-hint driver state so we can RESTORE
            # it on the OFF transition. Without this, ON->OFF was a
            # no-op write and left the camera dark until app restart
            # (dad + user reported). Guard with `if not _armed:` so
            # only the very first ON captures the true baseline; a
            # mid-armed re-open (recovery reopen, hot-swap) uses the
            # cap-identity guard below to invalidate the stash.
            try:
                pre_auto = cap.get(cv2.CAP_PROP_AUTO_EXPOSURE) if hasattr(cv2, "CAP_PROP_AUTO_EXPOSURE") else None
                pre_exp = cap.get(cv2.CAP_PROP_EXPOSURE) if hasattr(cv2, "CAP_PROP_EXPOSURE") else None
                pre_fps = cap.get(cv2.CAP_PROP_FPS) if hasattr(cv2, "CAP_PROP_FPS") else None
                pre_fourcc = int(cap.get(cv2.CAP_PROP_FOURCC)) if hasattr(cv2, "CAP_PROP_FOURCC") else None
                sys.stderr.write(
                    f"[r49-short-shutter] pre-hint driver state: "
                    f"backend={_backend_name!r} "
                    f"auto_exp={pre_auto} exp={pre_exp} "
                    f"driver_fps={pre_fps} fourcc={pre_fourcc}\n"
                )
                sys.stderr.flush()
            except Exception as _e:
                pre_auto = None
                pre_exp = None
                try:
                    sys.stderr.write(
                        f"[r49-short-shutter] pre-hint readback exception: "
                        f"{type(_e).__name__}: {_e}\n"
                    )
                    sys.stderr.flush()
                except Exception:
                    pass
            # r51: stash the true baseline. Only stash on the first
            # ON transition (guarded by _armed=False) AND only when
            # the cap identity matches or is a fresh cap. Cap identity
            # invalidation happens in the OFF/restore block below on
            # recovery reopen. Sentinels: -0.99 for AUTO (typical
            # "unset/unknown" is -1.0 on Realtek + Kiyo), -19.99 for
            # EXP (real IR/low-light UVCs advertise log2 down to -15).
            try:
                _armed = bool(getattr(self, "_r51_shutter_armed", False))
                _cap_id_stash = getattr(self, "_r51_shutter_cap_id", None)
                _cur_cap_id = id(cap)
                if _armed and _cap_id_stash != _cur_cap_id:
                    # Different cap object -> stash from a prior cap
                    # is meaningless. Reset armed state and let the
                    # `if not _armed` branch below re-stash.
                    self._r51_shutter_armed = False
                    _armed = False
                if not _armed:
                    # v1.1.7.1 (dad rig 2026-08-19): reject values that
                    # look like MANUAL mode. UVC drivers latch EXPOSURE
                    # at the device level, so a fresh session start can
                    # read back a MANUAL state that a prior session left
                    # behind (either our own r49 hint from a previous run
                    # of Touchless, or user's other camera software).
                    # If we stash that manual state as "baseline" and
                    # later "restore" it on FSS-off, we just re-apply
                    # the short-shutter darkness we were trying to escape.
                    # DSHOW manual AUTO = 0.25, auto = 0.75.
                    # MSMF/V4L2 manual AUTO = 1.0, auto = 3.0.
                    # Short-shutter EXP is in the -3 or more-negative
                    # range (log2 seconds); real auto-exposure values
                    # in indoor lighting are typically -2 to 0.
                    _auto_looks_manual = pre_auto is not None and (
                        abs(float(pre_auto) - 0.25) < 0.05
                        or abs(float(pre_auto) - 1.0) < 0.05
                    )
                    _exp_looks_short_shutter = (
                        pre_exp is not None
                        and float(pre_exp) < -2.5
                    )
                    _valid_auto = (
                        pre_auto is not None
                        and pre_auto > -0.99
                        and not _auto_looks_manual
                    )
                    _valid_exp = (
                        pre_exp is not None
                        and pre_exp > -19.99
                        and not _exp_looks_short_shutter
                    )
                    self._r51_shutter_stashed_auto = pre_auto if _valid_auto else None
                    self._r51_shutter_stashed_exp = pre_exp if _valid_exp else None
                    self._r51_shutter_cap_id = _cur_cap_id
                    self._r51_shutter_armed = True
                    try:
                        sys.stderr.write(
                            f"[r51-shutter-stash] pre_auto={pre_auto} "
                            f"pre_exp={pre_exp} "
                            f"auto_looks_manual={_auto_looks_manual} "
                            f"exp_looks_short_shutter={_exp_looks_short_shutter} "
                            f"-> stashed_auto={self._r51_shutter_stashed_auto} "
                            f"stashed_exp={self._r51_shutter_stashed_exp}\n"
                        )
                        sys.stderr.flush()
                    except Exception:
                        pass
            except Exception:
                pass
            ok_auto = None
            ok_exp = None
            if sys.platform == "darwin":
                ok_auto, ok_exp = self._apply_macos_short_shutter(cap)
            else:
                # r51: two independent try blocks so an AUTO failure
                # never suppresses the load-bearing EXPOSURE write. Per
                # our field-log analysis, EXPOSURE is what actually
                # changes fps on Realtek + Kiyo Pro; AUTO is a no-op on
                # both (driver rejects). Losing EXPOSURE because AUTO
                # threw on an unknown MSMF driver would defeat the whole
                # feature.
                try:
                    if hasattr(cv2, "CAP_PROP_AUTO_EXPOSURE"):
                        ok_auto = cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1.0)
                except Exception as _e:
                    try:
                        sys.stderr.write(
                            f"[r49-short-shutter] AUTO write exception: "
                            f"{type(_e).__name__}: {_e}\n"
                        )
                        sys.stderr.flush()
                    except Exception:
                        pass
                try:
                    if hasattr(cv2, "CAP_PROP_EXPOSURE"):
                        ok_exp = cap.set(cv2.CAP_PROP_EXPOSURE, -6.0)
                except Exception as _e:
                    try:
                        sys.stderr.write(
                            f"[r49-short-shutter] EXP write exception: "
                            f"{type(_e).__name__}: {_e}\n"
                        )
                        sys.stderr.flush()
                    except Exception:
                        pass
            # v1.1.7.9 display-invariance: flag the display path to lift
            # the dark short-shutter frame back to natural brightness.
            self._short_shutter_active_for_display = True
            try:
                post_auto = cap.get(cv2.CAP_PROP_AUTO_EXPOSURE) if hasattr(cv2, "CAP_PROP_AUTO_EXPOSURE") else None
                post_exp = cap.get(cv2.CAP_PROP_EXPOSURE) if hasattr(cv2, "CAP_PROP_EXPOSURE") else None
                post_fps = cap.get(cv2.CAP_PROP_FPS) if hasattr(cv2, "CAP_PROP_FPS") else None
                post_fourcc = int(cap.get(cv2.CAP_PROP_FOURCC)) if hasattr(cv2, "CAP_PROP_FOURCC") else None
                sys.stderr.write(
                    f"[r49-short-shutter] post-hint driver readback: "
                    f"backend={_backend_name!r} "
                    f"auto_exp_set_return={ok_auto} exp_set_return={ok_exp} "
                    f"auto_exp={post_auto} exp={post_exp} "
                    f"driver_fps={post_fps} fourcc={post_fourcc}\n"
                )
                sys.stderr.flush()
            except Exception as _e:
                try:
                    sys.stderr.write(
                        f"[r49-short-shutter] post-hint readback exception: "
                        f"{type(_e).__name__}: {_e}\n"
                    )
                    sys.stderr.flush()
                except Exception:
                    pass
            # r49 diag Q3 anchor: arm a bounded delivered-frame-dt
            # sample window so the next perf-monitor cycles can be
            # correlated to the hint. The tick loop already tracks
            # per-tick timing for self._fps; this flag just tells
            # a future dt-histogram consumer 'measurement starts
            # here'. Even without a histogram consumer, this line
            # gives the log a wall-time anchor to bisect against.
            try:
                import time as _t
                self._r49_hint_applied_wallclock = _t.time()
                # r50: reset the one-shot post_hint_fps guard so a
                # mid-session mode swap that re-fires this block
                # (Lite Mode toggle, Low FPS toggle) re-samples fps.
                self._r50_post_hint_fps_reported = False
                sys.stderr.write(
                    f"[r49-short-shutter] armed: hint_applied_ts={self._r49_hint_applied_wallclock:.3f} "
                    f"(watch next 6 perf-monitor lines for fps delta)\n"
                )
                sys.stderr.flush()
            except Exception:
                pass
            # r50: first-fire acknowledgement pill. Fires only when the
            # classifier auto-applied the hint (reason='classifier_generic')
            # AND this camera name has not yet had the pill shown. The
            # pill signal rides the existing engine_log Signal(str) — main
            # window intercepts the '[r50-classifier-pill]' prefix and
            # renders a 4s ProcessingOverlay toast + persists the camera
            # name into config.short_shutter_pill_shown_for_cameras.
            # Explicit user-config or env-override paths deliberately
            # skip the pill: those users made the choice themselves and
            # already know what's happening.
            try:
                _reason = getattr(self, "_r50_last_decision_reason", None)
                if _reason == "classifier_generic":
                    _display = getattr(self, "_r50_last_display_name", "") or ""
                    _display_key = _display.lower().strip()
                    shown = list(
                        getattr(
                            self.config,
                            "short_shutter_pill_shown_for_cameras",
                            [],
                        )
                    )
                    already_shown = _display_key in {str(x).lower().strip() for x in shown}
                    if _display_key and not already_shown:
                        engine_log = getattr(self, "engine_log", None)
                        if engine_log is not None:
                            try:
                                engine_log.emit(
                                    "[r50-classifier-pill] "
                                    "Your camera is set up for smooth tracking. "
                                    "You can change this in Settings > Camera anytime. "
                                    f"camera={_display}"
                                )
                            except Exception:
                                pass
            except Exception:
                pass
            # v1.1.7.1 Layer 3: auto-rollback if the hint made the
            # frame too dark. Applies ONLY when the classifier auto-
            # applied the hint on this camera (reason=classifier_generic).
            # User-explicit config (user_chose=True with cfg=True) and
            # env override are respected as-is — those users made an
            # informed choice and know the trade-off. For the auto-
            # applied case, we sample luma over ~300 ms after the driver
            # settles; if median falls below threshold, MediaPipe can't
            # detect hands regardless of how smooth the pipeline is, so
            # we restore the pre-hint state and mark the camera as
            # "auto-hint tried, rolled back" for the remainder of the
            # session.
            _auto_applied = _decision_reason == "classifier_generic"
            if _auto_applied and cap is not None:
                try:
                    # Settle time: the driver needs a beat to commit the
                    # new exposure before its readback matches what we
                    # just wrote. Sampling too early = we read the pre-
                    # write bright frames and mis-conclude "not dark."
                    time.sleep(0.15)
                    _luma_samples: list[float] = []
                    _luma_deadline = time.monotonic() + 0.35
                    while len(_luma_samples) < 10 and time.monotonic() < _luma_deadline:
                        try:
                            _ok, _frame = cap.read()
                        except Exception:
                            _ok, _frame = False, None
                        if not _ok or _frame is None:
                            try:
                                time.sleep(0.005)
                            except Exception:
                                pass
                            continue
                        try:
                            _small = _frame[::4, ::4]
                            if _small.ndim < 3 or _small.size == 0:
                                continue
                            _b = _small[:, :, 0].astype(np.float32)
                            _g = _small[:, :, 1].astype(np.float32)
                            _r = _small[:, :, 2].astype(np.float32)
                            _y = 0.0722 * _b + 0.7152 * _g + 0.2126 * _r
                            _luma_samples.append(float(np.median(_y)))
                        except Exception:
                            continue
                    if len(_luma_samples) >= 4:
                        _median_luma = float(np.median(_luma_samples))
                    else:
                        _median_luma = None
                    # Threshold 45 — slightly more permissive than the
                    # ffmpeg safety net's 50 because r49 short-shutter
                    # legitimately produces a slightly darker frame on
                    # cameras where it's supposed to work (r49 field-log
                    # confirmed on Realtek). Only very dark = rollback.
                    _too_dark = (
                        _median_luma is not None
                        and _median_luma < 45.0
                    )
                    try:
                        sys.stderr.write(
                            f"[r49-short-shutter] auto-hint luma verify: "
                            f"samples={len(_luma_samples)} median={_median_luma} "
                            f"too_dark={_too_dark}\n"
                        )
                        sys.stderr.flush()
                    except Exception:
                        pass
                    if _too_dark:
                        # v1.1.7.1 rollback path — always kick to AUTO,
                        # NEVER trust stashed values here. The stash may
                        # have captured a manual state from a prior
                        # session (UVC latching), and writing it back
                        # would just re-apply the darkness. Kicking to
                        # auto is what actually gets us out of manual
                        # mode. The driver's own auto-exposure logic
                        # then picks a sensible exposure — that's
                        # exactly what we want when the hint's own
                        # short shutter proved too aggressive.
                        try:
                            _rb_wrote_something = False
                            if hasattr(cv2, "CAP_PROP_AUTO_EXPOSURE"):
                                try:
                                    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)  # DSHOW auto
                                    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3.0)   # MSMF/V4L2 auto
                                    _rb_wrote_something = True
                                except Exception:
                                    pass
                            # v1.1.7.1 (dad rig): nudge EXPOSURE back
                            # to a medium value. Just kicking AUTO to
                            # auto isn't enough on drivers that keep
                            # the manually-set EXPOSURE value even in
                            # auto mode — the shutter stays short and
                            # the preview stays dark. -4 in log2s ≈
                            # 62 ms which is a reasonable brightness
                            # for indoor lighting; if the driver re-
                            # enters true auto it will readjust from
                            # there. We prefer a stashed EXPOSURE if
                            # it looks safe (>-2.5), otherwise we
                            # nudge to -4 as a safe default.
                            _rb_exp_stash = getattr(self, "_r51_shutter_stashed_exp", None)
                            _rb_exp_target = None
                            if (
                                _rb_exp_stash is not None
                                and _rb_exp_stash > -2.5
                                and _rb_exp_stash > -19.99
                            ):
                                _rb_exp_target = float(_rb_exp_stash)
                            else:
                                _rb_exp_target = -4.0
                            if hasattr(cv2, "CAP_PROP_EXPOSURE") and _rb_exp_target is not None:
                                try:
                                    cap.set(cv2.CAP_PROP_EXPOSURE, _rb_exp_target)
                                    _rb_wrote_something = True
                                except Exception:
                                    pass
                            # Suppress future auto-apply on this camera
                            # this session so the darkness doesn't come
                            # back on the next Lite Mode toggle etc.
                            _display_key_rb = (
                                str(getattr(self, "_r50_last_display_name", "") or "")
                                .lower().strip()
                            )
                            _rolled_back_set = getattr(
                                self, "_r49_auto_rollback_cameras", None
                            )
                            if _rolled_back_set is None:
                                _rolled_back_set = set()
                                self._r49_auto_rollback_cameras = _rolled_back_set
                            if _display_key_rb:
                                _rolled_back_set.add(_display_key_rb)
                            # Clear armed state — we just restored, no
                            # need for the OFF branch to try to
                            # restore again on the next toggle.
                            self._r51_shutter_armed = False
                            self._r51_shutter_stashed_auto = None
                            self._r51_shutter_stashed_exp = None
                            self._r51_shutter_cap_id = None
                            try:
                                sys.stderr.write(
                                    f"[r49-short-shutter] AUTO-ROLLBACK: "
                                    f"wrote_something={_rb_wrote_something} "
                                    f"rb_auto={_rb_auto} rb_exp={_rb_exp} "
                                    f"camera={_display_key_rb!r}. Auto-classifier "
                                    f"suppressed for this camera this session.\n"
                                )
                                sys.stderr.flush()
                            except Exception:
                                pass
                        except Exception as _e:
                            try:
                                sys.stderr.write(
                                    f"[r49-short-shutter] rollback exception: "
                                    f"{type(_e).__name__}: {_e}\n"
                                )
                                sys.stderr.flush()
                            except Exception:
                                pass
                except Exception:
                    pass
        else:
            # r51: OFF branch. If we previously armed the hint on
            # this cap, restore the stashed pre-hint AUTO_EXPOSURE +
            # EXPOSURE. Without this, ON->OFF was a no-op and the
            # camera stayed dark until app restart. Only clears the
            # armed flag on VERIFIED restore (post-readback matches
            # target within 0.5) so a silently-rejecting driver
            # doesn't lock the user into permanent darkness — the
            # next OFF click retries the same restore.
            #
            # r53 v7: also unconditionally kick the driver back into
            # AUTO_EXPOSURE mode even when not armed. This handles
            # the case where the camera FIRST opened with FSS=True
            # (persisted config), so the stash captured the manual
            # values as "baseline" and restoring them keeps the
            # camera dark. DSHOW auto=0.75, MSMF/V4L2 auto=3.0 —
            # write both since neither backend errors on an out-of-
            # range value, and one of them will be honoured.
            try:
                _armed = bool(getattr(self, "_r51_shutter_armed", False))
                _cap_id_stash = getattr(self, "_r51_shutter_cap_id", None)
                _cur_cap_id = id(cap)
                if _armed and _cap_id_stash != _cur_cap_id:
                    # Cap was reopened (recovery, hot-swap). The
                    # stashed values are stale. Drop armed state so
                    # a fresh ON re-captures a valid baseline.
                    self._r51_shutter_armed = False
                    self._r51_shutter_stashed_auto = None
                    self._r51_shutter_stashed_exp = None
                    self._r51_shutter_cap_id = None
                    _armed = False
            except Exception:
                _armed = False
            # v1.1.7.1: gate the auto-mode kick. r53 v7 wrote AUTO=0.75
            # then AUTO=3.0 on EVERY engine start — the intent was to
            # rescue cameras stuck in manual mode from a prior session.
            # In practice it also fired on cameras we had never touched
            # (HP HD Camera + other built-in laptop webcams whose names
            # don't match any classifier keyword), and on some of them
            # the double-write left the driver in a half-manual state
            # that produced a dark preview the moment the engine started
            # (Subodh Tope 2026-08-18 report: engine OFF = bright,
            # engine ON = dark). Only kick when there's evidence we
            # should be managing this camera's exposure:
            #   * _armed: we stashed pre-hint state in this session and
            #     may need to restore from auto (kick runs BEFORE the
            #     restore block below).
            #   * user_chose: user has explicitly toggled the checkbox,
            #     so we own this camera's exposure state either way.
            #   * classifier_verdict is True: this is a generic UVC we
            #     WOULD have applied the hint to; kick handles the case
            #     where a prior session persisted a manual state at the
            #     UVC device level and needs release.
            # For cameras where none of these apply (Kiyo Pro fresh
            # install, HP HD Camera fresh install, any built-in laptop
            # webcam that no user has ever configured), we leave the
            # driver's auto-exposure state exactly as it was.
            _user_chose_kick = bool(getattr(self.config, "camera_force_short_shutter_user_chose", False))
            # v1.1.7.1 (dad rig 2026-08-19): also kick when the driver's
            # CURRENT state looks like manual mode. This catches cameras
            # where a prior session (or unrelated software) left a manual
            # exposure state latched at the UVC device level and the
            # user hasn't explicitly asked for anything. Auto mode is
            # what 99% of users want on engine start — if we detect
            # the driver isn't in auto right now, kick it.
            # Only reads that clearly look manual trigger this — unknown
            # (-1) or ambiguous readings don't. Avoids kicking cameras
            # whose driver reports state opaquely.
            _pre_kick_auto = None
            try:
                if hasattr(cv2, "CAP_PROP_AUTO_EXPOSURE"):
                    _pre_kick_auto = cap.get(cv2.CAP_PROP_AUTO_EXPOSURE)
            except Exception:
                _pre_kick_auto = None
            _current_looks_manual = _pre_kick_auto is not None and (
                abs(float(_pre_kick_auto) - 0.25) < 0.05  # DSHOW manual
                or abs(float(_pre_kick_auto) - 1.0) < 0.05  # MSMF/V4L2 manual
            )
            # v1.1.7.3 (dad rig 2026-08-19): drivers that don't report
            # AUTO state truthfully (return -1.0 "unknown") were slipping
            # past _current_looks_manual and staying dark. Also look at
            # the read-back EXPOSURE: if it's in the short-shutter range
            # (< -2.5 log2s), the driver IS in a manual-short-shutter
            # state regardless of what AUTO reports. This is dad's rig
            # exactly: pre_kick_auto=-1.0, post_exp=-6.0, camera dark.
            _pre_kick_exp_readback = None
            try:
                if hasattr(cv2, "CAP_PROP_EXPOSURE"):
                    _pre_kick_exp_readback = cap.get(cv2.CAP_PROP_EXPOSURE)
            except Exception:
                _pre_kick_exp_readback = None
            _current_exp_looks_short = (
                _pre_kick_exp_readback is not None
                and _pre_kick_exp_readback > -19.99
                and _pre_kick_exp_readback < -2.5
            )
            _should_kick = (
                _armed
                or _user_chose_kick
                or _classifier_verdict is True
                or _current_looks_manual
                or _current_exp_looks_short
            )
            if _should_kick:
                try:
                    if hasattr(cv2, "CAP_PROP_AUTO_EXPOSURE"):
                        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)
                        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3.0)
                except Exception:
                    pass
                # v1.1.7.1 (dad rig): if EXPOSURE looks like a leftover
                # short-shutter latch, also nudge it back toward the
                # driver's default. Many DSHOW drivers keep the manual
                # EXPOSURE value even after AUTO_EXPOSURE=auto, so the
                # camera stays dark even in "auto" mode. Writing EXP=-4
                # (~62 ms shutter) gives the driver a bright starting
                # point; if it re-enters true auto, it will readjust
                # from there. Only fires when the read-back EXP is in
                # the short-shutter range (< -2.5 log2s = <180 ms).
                try:
                    _pre_kick_exp = cap.get(cv2.CAP_PROP_EXPOSURE) if hasattr(cv2, "CAP_PROP_EXPOSURE") else None
                except Exception:
                    _pre_kick_exp = None
                if (
                    _pre_kick_exp is not None
                    and _pre_kick_exp > -19.99
                    and _pre_kick_exp < -2.5
                ):
                    try:
                        cap.set(cv2.CAP_PROP_EXPOSURE, -4.0)
                    except Exception:
                        pass
                    # v1.1.7.9 display-invariance: -4 (62 ms shutter) is
                    # still shorter than the driver's typical auto pick
                    # (-2 / 250 ms). Enable display gamma-lift so the
                    # user's preview stays natural-looking even when the
                    # driver leaves EXP latched. If auto really does
                    # kick in and the sensor brightens, the compensation
                    # function's median-check returns a no-op anyway.
                    self._short_shutter_active_for_display = True
            # Post-kick readback for diagnosis.
            try:
                _kick_auto = cap.get(cv2.CAP_PROP_AUTO_EXPOSURE) if hasattr(cv2, "CAP_PROP_AUTO_EXPOSURE") else None
                _kick_exp = cap.get(cv2.CAP_PROP_EXPOSURE) if hasattr(cv2, "CAP_PROP_EXPOSURE") else None
                sys.stderr.write(
                    f"[r53-shutter-off-kick] armed={_armed} "
                    f"should_kick={_should_kick} "
                    f"user_chose={_user_chose_kick} "
                    f"verdict={_classifier_verdict} "
                    f"current_looks_manual={_current_looks_manual} "
                    f"pre_kick_auto={_pre_kick_auto} "
                    f"post_auto={_kick_auto} post_exp={_kick_exp}\n"
                )
                sys.stderr.flush()
            except Exception:
                pass
            if _armed:
                _target_auto = getattr(self, "_r51_shutter_stashed_auto", None)
                _target_exp = getattr(self, "_r51_shutter_stashed_exp", None)
                _restore_ok_auto = None
                _restore_ok_exp = None
                # Two independent try blocks — AUTO failure must not
                # suppress the load-bearing EXPOSURE restore.
                try:
                    if _target_auto is not None and hasattr(cv2, "CAP_PROP_AUTO_EXPOSURE"):
                        _restore_ok_auto = cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, float(_target_auto))
                except Exception:
                    pass
                try:
                    if _target_exp is not None and hasattr(cv2, "CAP_PROP_EXPOSURE"):
                        _restore_ok_exp = cap.set(cv2.CAP_PROP_EXPOSURE, float(_target_exp))
                except Exception:
                    pass
                # Post-restore readback + verification.
                _post_auto = None
                _post_exp = None
                try:
                    if hasattr(cv2, "CAP_PROP_AUTO_EXPOSURE"):
                        _post_auto = cap.get(cv2.CAP_PROP_AUTO_EXPOSURE)
                    if hasattr(cv2, "CAP_PROP_EXPOSURE"):
                        _post_exp = cap.get(cv2.CAP_PROP_EXPOSURE)
                except Exception:
                    pass
                _exp_ok = (
                    _target_exp is None
                    or (
                        _post_exp is not None
                        and abs(float(_post_exp) - float(_target_exp)) < 0.5
                    )
                )
                try:
                    sys.stderr.write(
                        f"[r51-short-shutter] restore: "
                        f"target_auto={_target_auto} target_exp={_target_exp} "
                        f"set_return_auto={_restore_ok_auto} set_return_exp={_restore_ok_exp} "
                        f"post_auto={_post_auto} post_exp={_post_exp} "
                        f"verified={_exp_ok}\n"
                    )
                    sys.stderr.flush()
                except Exception:
                    pass
                if _exp_ok:
                    # Verified restore -> clear the armed state so
                    # the next ON captures a fresh baseline.
                    self._r51_shutter_armed = False
                    self._r51_shutter_stashed_auto = None
                    self._r51_shutter_stashed_exp = None
                    self._r51_shutter_cap_id = None
                else:
                    # Driver silently rejected the restore. Keep
                    # armed=True + stash intact so the next OFF click
                    # retries against the same baseline. Never lock
                    # the user into permanent darkness.
                    try:
                        sys.stderr.write(
                            "[r51-short-shutter] restore rejected; "
                            "a camera reopen may be needed\n"
                        )
                        sys.stderr.flush()
                    except Exception:
                        pass

    def _restore_normal_capture_tuning(self, open_result) -> None:
        cap = None
        if isinstance(open_result, tuple) and len(open_result) >= 2:
            cap = open_result[1]
        elif open_result is not None and hasattr(open_result, "set"):
            cap = open_result
        if cap is None:
            return
        try:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            if hasattr(cv2, "CAP_PROP_FPS"):
                cap.set(cv2.CAP_PROP_FPS, 30)
        except Exception:
            pass

    def _prepare_runtime_frame(self, frame):
        if frame is None:
            return frame
        try:
            height, width = frame.shape[:2]
        except Exception:
            return frame
        if not self._low_fps_active or width <= 640 or height <= 0:
            return frame
        scaled_height = max(1, int(round(height * (640.0 / float(width)))))
        try:
            return cv2.resize(frame, (640, scaled_height), interpolation=cv2.INTER_AREA)
        except Exception:
            return frame

    def _apply_brightness_boost_if_needed(self, frame):
        """v1.1.7.8: CLAHE-based low-light enhancement.

        Uses Contrast Limited Adaptive Histogram Equalization (CLAHE)
        on the Y channel of YCrCb color space. This is the exact
        technique Zoom, Google Meet, and Microsoft Teams use for their
        "adjust for low light" mode:

          * Luminance-only enhancement — the Y channel gets equalized
            while Cr and Cb (color difference) channels are untouched.
            Result: hue is preserved perfectly, ZERO yellow cast
            (fixed the 1.1.7.4-6 complaint that linear multiply
            amplified the camera's warm tint).
          * Adaptive per 8×8 tile — dark corners get more brightening
            than already-bright zones. Prevents blown-out highlights
            and preserves the overall look of the scene.
          * Hysteresis (enable at Y<70, disable at Y>100) — prevents
            the "adjusting" flicker the user reported on 1.1.7.4 when
            a linear boost turned on/off at a single threshold.
          * clipLimit=2.5 default — Zoom/Meet use similar values.
            Higher values (up to 5.0 via config) give more aggressive
            enhancement for genuinely dim scenes.

        Config knobs:
          - camera_brightness_boost_enabled (bool, default True) —
            master switch for the display path. False = raw pixels
            always. Detection path is unaffected (MediaPipe always
            benefits from the enhancement when dark).
          - camera_brightness_boost (int, default 0):
              0 = auto (fires when Y median < 70, off when > 100)
              1–100 = forced clipLimit scale (2.0 → 5.0)

        Cost: ~1.5 ms per 640×480 frame when enhancement fires;
        zero cost when disabled (early return). Luma sample runs
        every 15 ticks on a 4×-downsampled slice.
        """
        if frame is None:
            return frame
        try:
            cfg_boost = int(getattr(self.config, "camera_brightness_boost", 0) or 0)
        except Exception:
            cfg_boost = 0
        cfg_boost = max(0, min(100, cfg_boost))

        # Sample luma periodically. Every 15 ticks (~500 ms at 30 fps)
        # keeps the cost trivial while still reacting to lighting
        # changes fast enough that a user turning on a light doesn't
        # sit in over-boosted mode for long.
        _cnt = int(getattr(self, "_bboost_tick_counter", 0)) + 1
        self._bboost_tick_counter = _cnt
        _should_sample = (_cnt % 15) == 0 or not hasattr(self, "_bboost_last_luma")
        if _should_sample:
            try:
                small = frame[::4, ::4]
                if small.ndim >= 3 and small.size > 0:
                    b = small[:, :, 0].astype(np.float32)
                    g = small[:, :, 1].astype(np.float32)
                    r = small[:, :, 2].astype(np.float32)
                    y = 0.0722 * b + 0.7152 * g + 0.2126 * r
                    luma = float(np.median(y))
                    self._bboost_last_luma = luma
                    # Hysteresis: enable at Y<70, disable at Y>100.
                    # The 30-unit dead zone means a scene hovering
                    # around Y=80 doesn't toggle the state on every
                    # sample — the flicker complaint from 1.1.7.4
                    # was caused by a single-threshold trigger that
                    # kept flipping.
                    was_active = getattr(self, "_bboost_active", False)
                    if was_active:
                        self._bboost_active = luma < 100.0
                    else:
                        self._bboost_active = luma < 70.0
            except Exception:
                pass

        # Decide whether to apply enhancement.
        auto_active = getattr(self, "_bboost_active", False)
        if cfg_boost > 0:
            # Manual force: user picked a strength.
            clip_limit = 2.0 + (cfg_boost / 100.0) * 3.0  # 2.0 → 5.0
        elif auto_active:
            # Auto-mode: gentle default.
            clip_limit = 2.5
        else:
            return frame  # Nothing to do; frame is bright enough.

        # Apply CLAHE on Y channel — the whole point of this pipeline.
        # tileGridSize=8x8 gives good local adaptation on 640×480 without
        # visible tile boundaries. Larger tiles look global; smaller
        # tiles start showing tile-edge artifacts on flat regions.
        try:
            ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
            y_ch, cr, cb = cv2.split(ycrcb)
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
            y_boosted = clahe.apply(y_ch)
            ycrcb_boosted = cv2.merge([y_boosted, cr, cb])
            return cv2.cvtColor(ycrcb_boosted, cv2.COLOR_YCrCb2BGR)
        except Exception:
            return frame

    def _compensate_short_shutter_for_display(self, frame):
        """v1.1.7.9: display-side invariance for Boost Performance mode.

        The pipeline contract the user asked for: when Boost mode forces
        the camera to a short shutter (higher fps but darker sensor
        output), the DISPLAY should look the same as it would at the
        camera's natural exposure — the user sees the fps benefit
        without any visible dimming. When Boost is off, this function
        is a no-op (raw pass-through) so the user still sees the
        sensor's natural output.

        Method: adaptive gamma correction on the Y channel of YCrCb,
        targeting median luma ~= 100 (comfortable mid-range that reads
        as "normal room lighting, not dim"). Chosen over linear gain
        because gamma lifts midtones without clipping highlights (linear
        multiply on a dim frame produces the yellow cast we saw in
        1.1.7.4). Chosen over CLAHE because the target here is GLOBAL
        brightness restoration, not local contrast enhancement — CLAHE
        would over-enhance already-lit regions.

        Only fires when `self._short_shutter_active_for_display` is
        True, which the camera-open path sets whenever it actually
        writes EXPOSURE=-6 (or a similar short-shutter override) to the
        driver. If the driver silently rejects the write (r49 field
        note documents this on some UVCs), the sensor output is bright
        anyway and the median-check inside will short-circuit the
        compensation.

        Cost: cv2.LUT is O(N) over pixels, ~0.5 ms per 640x480 frame;
        one cvtColor round-trip is ~0.5 ms. Total ~1.2 ms per frame
        while Boost is active. Zero cost when Boost is off (early
        return before any cv2 call). Median sampled every 15 frames
        on a 4x-downsampled slice; LUT rebuilt only when the gamma
        target moves by >0.02 so lighting changes cost ~0 in steady
        state.
        """
        if frame is None:
            return frame
        if not getattr(self, "_short_shutter_active_for_display", False):
            return frame
        _cnt = int(getattr(self, "_ss_disp_tick", 0)) + 1
        self._ss_disp_tick = _cnt
        _sample_ok = _cnt % 15 == 0 or not hasattr(self, "_ss_disp_target_gamma")
        if _sample_ok:
            try:
                small = frame[::4, ::4]
                if small.ndim >= 3 and small.size > 0:
                    b = small[:, :, 0].astype(np.float32)
                    g = small[:, :, 1].astype(np.float32)
                    r = small[:, :, 2].astype(np.float32)
                    y = 0.0722 * b + 0.7152 * g + 0.2126 * r
                    _median = float(np.median(y))
                else:
                    _median = 128.0
            except Exception:
                _median = 128.0
            _TARGET = 100.0
            if _median >= _TARGET or _median < 2.0:
                self._ss_disp_target_gamma = None
            else:
                try:
                    gamma = float(
                        np.log(_TARGET / 255.0) / np.log(_median / 255.0)
                    )
                except Exception:
                    gamma = 1.0
                self._ss_disp_target_gamma = float(np.clip(gamma, 0.35, 1.0))
        _gamma = getattr(self, "_ss_disp_target_gamma", None)
        if _gamma is None:
            return frame
        _cached = getattr(self, "_ss_disp_lut_gamma", None)
        if _cached is None or abs(_cached - _gamma) > 0.02:
            try:
                idx = np.arange(256, dtype=np.float32) / 255.0
                self._ss_disp_lut = np.clip(
                    np.power(idx, _gamma) * 255.0, 0, 255
                ).astype(np.uint8)
                self._ss_disp_lut_gamma = _gamma
            except Exception:
                return frame
        try:
            ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
            y_ch, cr, cb = cv2.split(ycrcb)
            y_lifted = cv2.LUT(y_ch, self._ss_disp_lut)
            ycrcb_lifted = cv2.merge([y_lifted, cr, cb])
            return cv2.cvtColor(ycrcb_lifted, cv2.COLOR_YCrCb2BGR)
        except Exception:
            return frame

    def _tick(self) -> None:
        if not self._running or self._cap is None or self.engine is None:
            return
        self._tick_call_count += 1
        # Dead-cap auto-recovery. The threaded reader marks the
        # capture dead (isOpened() → False) when it hits the
        # consecutive-failure ceiling — typically a USB hiccup or
        # another app briefly grabbing the camera. Reopen the same
        # camera (rate-limited; capped at _camera_recovery_max_attempts)
        # so a transient drop doesn't permanently freeze the live
        # view on its last good frame.
        try:
            cap_alive = bool(self._cap.isOpened())
        except Exception:
            cap_alive = False
        if not cap_alive:
            self._attempt_camera_recovery()
            return
        # No-first-frame timeout. If the camera opened but never
        # actually produced a clean frame within ~5 s, surface a
        # one-time status update so the user gets visible feedback
        # instead of "Starting camera…" hanging silently.
        if not self._camera_first_frame_received and not self._camera_no_first_frame_warned:
            elapsed = time.monotonic() - self._camera_start_at
            if elapsed >= 5.0:
                self._camera_no_first_frame_warned = True
                self._emit_status("camera opened but no frames yet")
        # Per-frame timing diagnostic â€” sampled when Lite Mode is on
        # so we can attribute fps drops to camera vs MediaPipe vs
        # downstream work. Lazy so non-debug callers pay no
        # clock-syscall cost.
        debug_timing = self._perf_optimisations_enabled() or self._perf_log_enabled
        t0 = time.perf_counter() if debug_timing else 0.0
        ok, frame = self._cap.read()
        t_read = time.perf_counter() if debug_timing else 0.0
        # PERF DIAG (HGR_PERF_LOG=1): disambiguate the fps cap. Logs how often
        # _tick actually fires vs how often a genuinely NEW camera frame
        # arrives. tick~60/newframe~15 => camera is delivering ~15 fps (auto-
        # exposure/lighting); tick~15 => event-loop/timer throttled; tick~60 &
        # newframe~60 but low self._fps => runner back-pressure.
        if self._perf_log_enabled:
            self._tickdiag_ticks = getattr(self, "_tickdiag_ticks", 0) + 1
            _cap_ts = float(
                getattr(self._cap, "_latest_frame_ts", 0.0)
                or getattr(self._cap, "_last_consumed_ts", 0.0)
                or 0.0
            )
            if _cap_ts != getattr(self, "_tickdiag_last_ts", -1.0):
                self._tickdiag_newframes = getattr(self, "_tickdiag_newframes", 0) + 1
                self._tickdiag_last_ts = _cap_ts
            _dnow = time.monotonic()
            if not hasattr(self, "_tickdiag_at"):
                self._tickdiag_at = _dnow
            elif _dnow - self._tickdiag_at >= 2.0:
                _dt = _dnow - self._tickdiag_at
                try:
                    sys.stderr.write(
                        f"[tickdiag] tick={self._tickdiag_ticks / _dt:.1f}/s "
                        f"newframe={getattr(self, '_tickdiag_newframes', 0) / _dt:.1f}/s "
                        f"busy={self._engine_runner.busy} pending={self._async_result_pending}\n"
                    )
                    sys.stderr.flush()
                except Exception:
                    pass
                self._tickdiag_ticks = 0
                self._tickdiag_newframes = 0
                self._tickdiag_at = _dnow
        if not ok:
            # Camera stalled briefly (ffmpeg pipe between frames,
            # No fresh frame this poll â€” let the periodic 15 ms
            # QTimer drive the next attempt. Tight singleShot(0)
            # rescheduling here previously starved Qt's paint
            # event dispatcher on the main thread (paint rate
            # collapsed to ~2 fps even though the worker fired
            # emits at 30+ fps), because the event loop never got
            # enough idle time between ticks to process the
            # queued paint events.
            return
        # First clean frame of this session — flip the live status
        # from "Starting camera…" to "Touchless active" so the user
        # gets clear feedback that the engine actually came up.
        # `_camera_recovery_attempts` resets here too so a future
        # mid-session drop gets its own independent retry budget.
        if not self._camera_first_frame_received:
            self._camera_first_frame_received = True
            self._camera_recovery_attempts = 0
            self._camera_no_first_frame_warned = False
            try:
                self._emit_status("Touchless active")
                self.command_detected.emit("Gesture and voice control active")
            except Exception:
                pass
        # Always mirror to selfie view. The earlier
        # `camera_source_is_mirrored` toggle was meant for phone-camera
        # sources whose host app pre-mirrored the feed (e.g., Iriun
        # with its own mirror checkbox on), but in practice users
        # toggled it on for unrelated reasons and ended up with the
        # main app showing camera-perspective with no obvious cause.
        # The flag is kept on AppConfig for backwards compatibility
        # but ignored here so every Touchless surface â€” engine,
        # tutorial, recorder, sandbox, preview â€” shows the same
        # selfie view consistently.
        if not bool(getattr(self.config, "camera_source_is_mirrored", False)):
            frame = cv2.flip(frame, 1)
        frame = self._prepare_runtime_frame(frame)
        # Split display from detection — the pipeline contract.
        # Detection: CLAHE-preprocessed for MediaPipe accuracy in dim
        # scenes; MediaPipe never sees the raw dim frame.
        # Display: pristine sensor output BY DEFAULT (boost mode off).
        # When boost mode has forced a short shutter for fps, the
        # sensor output is darker than natural auto-exposure would be —
        # `_compensate_short_shutter_for_display` lifts it back with an
        # adaptive gamma curve on the Y channel so the user's preview
        # looks approximately the same whether boost is on or off.
        # This is the invariance contract: boost mode's ONLY visible
        # effect is higher fps; nothing else changes.
        detection_frame = self._apply_brightness_boost_if_needed(frame)
        t_prep = time.perf_counter() if debug_timing else 0.0
        # Camera-target drawing composite. The drawing canvas
        # accumulates strokes in _update_camera_drawing_canvas
        # (called per tick), but the GPU-decoupled-display rewrite
        # accidentally dropped the call site that BLITS that canvas
        # onto the displayed frame â€” so strokes were going into the
        # canvas correctly but never reached the live view, which
        # is why 'switch view' to camera mode showed nothing as the
        # user drew. Blend before the raw_frame_ready emit so the
        # mini viewer + enlarged live view (both painting from
        # raw_frame_ready) show the strokes the moment they're
        # drawn. Fast-paths to a no-op when target != "camera" or
        # the canvas is empty, so the screen-target case stays
        # zero-cost.
        try:
            self._blend_camera_drawing_overlay(frame)
        except Exception:
            pass
        # v1.1.7.9 display-invariance: apply the boost-mode gamma lift
        # only when the camera-open path flagged short-shutter as
        # active. When boost is off, this is a no-op fast path and the
        # emitted buffer is the raw frame.
        display_frame = self._compensate_short_shutter_for_display(frame)
        # Decoupled display path. CRUCIAL ordering: emit the raw
        # frame BEFORE the back-pressure check below. This is what
        # makes the live view update at camera fps even when GPU
        # inference is slow (Valorant or screen capture loading the
        # GPU pushes ONNX inference latency from ~30 ms to ~500 ms).
        # If we ran the back-pressure check first and returned early,
        # the display would only refresh when inference finishes â€”
        # collapsing visible fps to 2-5 and producing the multi-
        # second perceived lag users reported during gaming.
        try:
            capture_ts = float(getattr(self._cap, "_last_consumed_ts", 0.0) or 0.0)
            if capture_ts <= 0.0:
                capture_ts = time.monotonic()
            # r54 v3 REVERTED: display WB "fix" made colors worse in
            # practice — gray-world assumption failed on a scene with
            # non-neutral objects, and my correction was pushing an
            # already-broken frame further off. Root cause of the
            # green/purple cast is being properly diagnosed via
            # workflow. Emitting the raw frame like before.
            self.raw_frame_ready.emit(display_frame, capture_ts)
            self._raw_emit_count += 1
            self._note_display_fps()
        except Exception:
            pass
        # C14 diagnostic: raw emit rate + tick fire rate. Camera
        # reader delivers frames independently of engine work, so
        # the emit rate tells us whether QTimer is firing at its
        # 15 ms cadence (≈66 fps ceiling) or whether main-thread
        # work is pushing tick fires apart.
        if self._perf_optimisations_enabled():
            _now_perf = time.monotonic()
            if self._raw_emit_last_log <= 0.0:
                self._raw_emit_last_log = _now_perf
            elif (_now_perf - self._raw_emit_last_log) >= 2.0:
                _elapsed = _now_perf - self._raw_emit_last_log
                _emit_fps = self._raw_emit_count / _elapsed if _elapsed > 0 else 0.0
                _tick_fps = self._tick_call_count / _elapsed if _elapsed > 0 else 0.0
                try:
                    sys.stderr.write(
                        f"[raw_frame] emit rate: {_emit_fps:.1f} fps "
                        f"(tick fires: {_tick_fps:.1f} fps, {self._tick_call_count} "
                        f"calls / {_elapsed:.2f} s)\n"
                    )
                    sys.stderr.flush()
                except Exception:
                    pass
                self._raw_emit_count = 0
                self._tick_call_count = 0
                self._raw_emit_last_log = _now_perf
        # C22 diagnostic: system resource snapshot every ~5 s. Runs
        # regardless of perf-mode so the user can watch CPU / GPU
        # move as they toggle modes without needing Task Manager open
        # side-by-side. Cheap: psutil calls are microseconds; the
        # nvidia-smi shell out is bounded at 500 ms per query and
        # cached to a background thread so the main tick doesn't
        # jitter when the subprocess is slow.
        self._maybe_log_perf_monitor()
        # Back-pressure: if the engine runner is still chewing on the
        # previous frame, drop the rest of this tick (no inference,
        # no debug payload). Display already went out above so the
        # live view stays smooth. The next tick will pick up a fresh
        # camera frame for inference once the runner is free.
        # Runner.submit() also self-checks _busy, so the request queue
        # stays at depth=1 either way.
        #
        # Note: we deliberately no longer gate on
        # _async_result_pending here. The runner clears its _busy
        # flag the instant inference finishes (before it emits the
        # queued result signal), so gating on _busy alone lets the
        # next tick submit a fresh frame the moment the runner is
        # idle, without waiting the extra ~5-15 ms it takes for Qt's
        # queued-signal delivery to land _on_engine_result on the
        # main thread. That single change lifts the tick cadence
        # from ~25 fps to camera-limited (see release notes for
        # measurements). _last_result_had_hand may be one cycle stale
        # for the skip-inference optimization; that only affects when
        # we can shortcut the neutral-frame case, never correctness.
        if self._engine_runner.busy:
            return
        tick_now = time.time()
        if self._should_skip_forced_fps_tick(tick_now):
            return
        # Pipeline freeze: the recorder dialog is open. Stop here â€”
        # the recorder runs its OWN MediaPipe pass on the raw frame
        # we just emitted, so a second pass in this worker would be
        # pure duplicate cost (and was the cause of the user-reported
        # "camera lags while custom-gesture window is open" stutter).
        # Gesture actions and debug-payload emits are also short-
        # circuited so the user can intentionally hold a sample pose
        # without the worker firing the matching binding.
        if self._frozen:
            return
        # Smart skip-frame inference: when the previous frame was
        # empty (no hand), skip MediaPipe this tick without running
        # the result pipeline. Always on — not just Lite/GPU — because
        # empty-frame MediaPipe was the random live-view hitch with
        # nobody in frame. Full-rate inference resumes the instant a
        # hand is tracked, so swipes / repeat-circle keep every
        # landmark sample. Empty frames skip every other tick so idle
        # GPU cost drops without starving hand reacquisition.
        skip_inference, self._idle_skip_ticks = self._empty_frame_skip_inference(
            self._last_result_had_hand, self._idle_skip_ticks
        )
        # Stash timing context so _on_engine_result can finish the
        # debug breakdown using the same per-tick start markers, even
        # when the engine call returns asynchronously from the runner
        # thread.
        self._tick_timing_state = (debug_timing, t0, t_read, t_prep)
        if skip_inference:
            # Empty-frame skip used to synthesise a no-hand result and
            # still run the full _on_engine_result pipeline (YouTube
            # ticks, custom hand_lost, overlay, FPS). That GUI-thread
            # work was the remaining idle hitch in GPU mode. Display
            # already went out via raw_frame_ready above; the last
            # real empty inference already cleared the skeleton.
            self._inference_skipped_last = True
            return
        self._inference_skipped_last = False
        # Push left-handed mode to the engine each tick (cheap bool set
        # on the shared engine object the runner also holds). Read live
        # from config so the Settings toggle takes effect next frame.
        if self.engine is not None:
            self.engine.swap_handedness = bool(getattr(self.config, "left_handed_mode", False))
        # v1.1.7.5: submit the DETECTION frame (may be brightness-boosted)
        # to MediaPipe. The display already got the pristine frame via
        # raw_frame_ready above. This is the split-frame architecture:
        # detection runs on whatever gets MediaPipe the best signal,
        # while the user sees an unmodified camera picture.
        if self._engine_runner.is_running:
            self._async_result_pending = True
            if self._engine_runner.submit(detection_frame):
                return
            self._async_result_pending = False
        try:
            result = self.engine.process_frame(detection_frame)
        except Exception:
            traceback.print_exc()
            return
        try:
            self._attach_custom_hands_on_engine_thread(detection_frame, result)
        except Exception:
            pass
        self._on_engine_result(detection_frame, result)

    def _on_engine_result(self, frame, result) -> None:
        # Runs on the GUI (main) thread. Reached via two paths:
        #   1. Direct call from _tick when the runner couldn't accept
        #      a submission and we fell back to inline inference.
        #   2. Queued signal from the _EngineRunner thread when async
        #      inference completes â€” Qt's auto-connection routes the
        #      cross-thread emit through the GUI thread's event loop,
        #      so widget mutations below are safe.
        # Empty-frame skip returns from _tick without coming here.
        # Clear the async-pending guard up front so the next _tick is
        # free to dispatch even if the bail-outs below trigger. The
        # skip-inference / inline-fallback paths set this flag to
        # False before calling us, so this is a no-op for them.
        self._async_result_pending = False
        # If the worker stopped between dispatch and result delivery
        # (Stop button hit while a frame was in flight), drop the
        # result silently â€” the receivers have already detached.
        if not self._running or result is None:
            return
        debug_timing, t0, t_read, t_prep = self._tick_timing_state or (False, 0.0, 0.0, 0.0)
        # Always reflect the current result's hand-presence in
        # _last_result_had_hand. neutral_result_for_frame always
        # reports found=False, so the skip-frame state machine still
        # behaves identically to the pre-async implementation.
        self._last_result_had_hand = bool(result.found)
        # YouTube auto-pause-on-absence + auto-skip-ads background
        # ticks. Driven off the per-frame callback because we already
        # have wall-clock time + hand presence here; no separate
        # QTimer needed. Both features short-circuit when their
        # respective settings are off.
        try:
            self._youtube_absence_tick(bool(result.found))
        except Exception:
            pass
        try:
            self._youtube_auto_skip_tick()
        except Exception:
            pass
        t_engine = time.perf_counter() if debug_timing else 0.0
        self._drawing_secondary_hand_reading = getattr(result, "secondary_hand_reading", None)
        now = time.time()
        self._refresh_fullscreen_foreground(now)
        self._maybe_auto_toggle_low_fps(now)
        self._maybe_offer_low_fps_suggestion(now)
        self._drain_voice_results()
        if not self._dictation_active:
            try:
                self.text_input_controller.remember_active_window()
            except Exception:
                pass
        monotonic_now = time.monotonic()
        result = self._normalize_result_right_primary(result)
        # Emit per-hand info for GPU-side overlay rendering.
        # Receivers feed these into GpuVideoWidget.update_landmarks
        # which draws bbox + handedness/gesture banner + connections
        # + joints via the Qt paint engine, replacing the CPU cv2
        # draw_hand_overlay path that used to mutate the display
        # BGR buffer every frame.
        #
        # Both hands are treated equally: each hand's box is red by
        # default and turns green when THAT hand's own prediction
        # is non-neutral. There is no primary/secondary distinction
        # in the display â€” _normalize_result_right_primary above
        # only re-routes which hand the existing right-hand-specific
        # internal logic operates on; the engine produces a
        # prediction for each hand independently (result.prediction
        # for tracked_hand, result.secondary_prediction for
        # secondary_tracked_hand).
        try:
            # Pull the runner's currently-held custom gesture (if any)
            # so we can paint its name as the banner over the matching
            # hand â€” same affordance built-in gestures get. Resolved
            # ONCE per frame; applied to whichever hand's handedness
            # matches the gesture's stored hand. None when no match.
            custom_match: Optional[tuple[str, Optional[str]]] = None
            if self._custom_gesture_runner is not None:
                try:
                    custom_match = self._custom_gesture_runner.current_match
                except Exception:
                    custom_match = None
            # Dynamic-gesture overlay: show the gesture name briefly
            # after it fires. Static-runner takes priority — it has a
            # held state, so its label reflects an in-progress gesture.
            if custom_match is None:
                dyn = getattr(self, "_dynamic_gesture_runtime", None)
                if dyn is not None:
                    try:
                        custom_match = dyn.current_match(time.monotonic())
                    except Exception:
                        custom_match = None
            if custom_match is None:
                seq = getattr(self, "_pose_sequence_runtime", None)
                if seq is not None:
                    try:
                        custom_match = seq.current_banner()
                    except Exception:
                        custom_match = None

            def _apply_custom_label(default_label: str, default_active: bool, hand_handedness: Optional[str]):
                """Override the banner with the custom gesture's name
                when it's the gesture being held on this hand."""
                if custom_match is None:
                    return default_label, default_active
                cm_name, cm_hand = custom_match
                # cm_hand=None means the gesture is bound to either
                # hand â€” show on whichever hand is matched. Otherwise
                # only show on the matching hand.
                if cm_hand is not None and cm_hand != hand_handedness:
                    return default_label, default_active
                return cm_name, True

            hands_info: list = []
            if result.found and result.tracked_hand is not None:
                label, active = self._derive_display_label(
                    result.prediction, getattr(result, "hand_reading", None)
                )
                primary_handedness = getattr(result.tracked_hand, "handedness", None)
                label, active = self._filter_banner_label_by_handedness(
                    label, active, primary_handedness
                )
                label, active = self._hide_unnamed_recognizer_label(label, active)
                label, active = _apply_custom_label(label, active, primary_handedness)
                # Display: underscore -> space so derived labels read
                # naturally ('three together', 'four together')
                # instead of with underscores. Done after the filter
                # pipeline so any binding lookup that needs the
                # underscore form still sees it.
                display_label = label.replace("_", " ") if label else label
                hands_info.append(
                    self._build_hand_overlay_info(
                        result.tracked_hand,
                        label=display_label,
                        active=active,
                    )
                )
            secondary_hand = getattr(result, "secondary_tracked_hand", None)
            if secondary_hand is not None:
                sec_pred = getattr(result, "secondary_prediction", None)
                label, active = self._derive_display_label(
                    sec_pred, getattr(result, "secondary_hand_reading", None)
                )
                sec_handedness = getattr(secondary_hand, "handedness", None)
                label, active = self._filter_banner_label_by_handedness(
                    label, active, sec_handedness
                )
                label, active = self._hide_unnamed_recognizer_label(label, active)
                label, active = _apply_custom_label(label, active, sec_handedness)
                display_label = label.replace("_", " ") if label else label
                hands_info.append(
                    self._build_hand_overlay_info(
                        secondary_hand,
                        label=display_label,
                        active=active,
                    )
                )
            mouse_overlay = None
            # Hand-selector dialogs (pen options, eraser options,
            # monitor selection, etc.) also use the red mouse-control
            # box for cursor mapping, with a sensitivity boost. Render
            # the SAME visual red box here so the user can see and
            # use the ergonomic hand-reach region during the dialog
            # exactly like they do in mouse mode.
            if self._utility_capture_selection_active and self._utility_capture_box_bounds is not None:
                mouse_overlay = {
                    "bounds": tuple(float(v) for v in self._utility_capture_box_bounds),
                    "anchor": None,
                    "cursor": (
                        tuple(float(v) for v in self._utility_capture_cursor_norm)
                        if self._utility_capture_cursor_norm is not None
                        else None
                    ),
                    "active_monitor_index": None,
                }
            elif self._mouse_mode_enabled:
                try:
                    debug = self.mouse_tracker.debug_state
                except Exception:
                    debug = None
                if debug is not None and debug.camera_control_bounds is not None:
                    raw_active = getattr(self.config, "mouse_active_monitor_index", None)
                    mouse_overlay = {
                        "bounds": tuple(float(v) for v in debug.camera_control_bounds),
                        "anchor": (
                            tuple(float(v) for v in debug.camera_anchor_position)
                            if debug.camera_anchor_position is not None
                            else None
                        ),
                        # Virtual cursor position in [0, 1] across
                        # the full virtual desktop. Lets the live
                        # view widget render a moving dot inside the
                        # red box so the user sees the cursor
                        # responding to their hand even when their
                        # eyes are on the camera feed instead of the
                        # actual mouse pointer.
                        "cursor": (
                            tuple(float(v) for v in debug.cursor_position)
                            if debug.cursor_position is not None
                            else None
                        ),
                        # Which monitor the cursor is currently
                        # constrained to (None = all monitors). The
                        # gpu_video_widget overlay paints the chosen
                        # screen highlighted and dims the rest so the
                        # user can confirm at a glance which display
                        # they're driving — fixes the "I picked
                        # Monitor 1 but the live view still shows
                        # both highlighted" report.
                        "active_monitor_index": (
                            int(raw_active) if isinstance(raw_active, int) else None
                        ),
                    }
            # In switch-view (camera-target) drawing mode, hide the hand
            # skeleton / bbox / gesture-label overlay the live-view widget
            # paints on top — the user wants their DRAWING (baked into the
            # frame, with the fingertip cursor) to be the top layer, not the
            # "camera reading my hand" graphics. The drawing + cursor still
            # show because they're in the frame, not in this overlay.
            hide_hand_overlay = bool(
                self._drawing_mode_enabled and self._drawing_render_target == "camera"
            )
            # C13: elide the emit when the payload would be a
            # steady-state repeat (no hands, no mouse overlay, no
            # drawing-mode override). The hand-present-to-absent
            # transition still fires because _last_emit_had_hands is
            # True on that specific cycle, forcing should_emit to True
            # and letting the widget clear its stashed skeleton.
            has_hands = bool(hands_info)
            should_emit = True
            if (
                not has_hands
                and mouse_overlay is None
                and not hide_hand_overlay
                and not self._last_emit_had_hands
            ):
                should_emit = False
            if should_emit:
                if mouse_overlay is not None or hide_hand_overlay:
                    payload = {
                        "hands": hands_info,
                        "mouse_overlay": mouse_overlay,
                        "hide_hand_overlay": hide_hand_overlay,
                    }
                else:
                    payload = hands_info
                self.engine_landmarks_ready.emit(payload)
                self._last_emit_had_hands = has_hands
        except Exception:
            pass

        # Custom-gesture live processing (static, dynamic, pose
        # sequence). Recorders use MediaPipe Hands at complexity=1;
        # GPU/Lite engine landmarks are a different distribution, so
        # one shared hand list per tick is either engine output (CPU
        # MediaPipe complexity 1) or a private MediaPipe pass.
        fired_dyn = None
        custom_hands: list = []
        custom_hands_sampled = False
        runner_now = time.monotonic()
        dynamic_runtime = getattr(self, "_dynamic_gesture_runtime", None)
        seq_runtime = getattr(self, "_pose_sequence_runtime", None)
        try:
            if self._custom_gesture_runner is not None:
                self._custom_gesture_runner.maybe_reload_if_changed(runner_now)
            if dynamic_runtime is not None:
                dynamic_runtime.maybe_reload_if_changed(runner_now)
            if seq_runtime is not None:
                seq_runtime.maybe_reload_if_changed(runner_now)
        except Exception:
            pass

        need_custom = False
        try:
            if self._custom_gesture_runner is not None and self._custom_gesture_runner.has_gestures:
                need_custom = True
            if dynamic_runtime is not None and dynamic_runtime.has_dynamic_gestures():
                need_custom = True
            if seq_runtime is not None and seq_runtime.has_sequences():
                need_custom = True
        except Exception:
            need_custom = bool(self._custom_gesture_runner is not None)

        if need_custom:
            try:
                custom_hands, custom_hands_sampled = self._custom_hands_this_frame(
                    result, frame
                )
            except Exception as exc:
                print(f"[custom-gestures] landmark extract error: {exc}")
                custom_hands, custom_hands_sampled = [], True

        if self._custom_gesture_runner is not None:
            try:
                fired = None
                if custom_hands:
                    fired = self._custom_gesture_runner.process_engine_hands(
                        custom_hands, runner_now
                    )
                elif custom_hands_sampled and need_custom:
                    self._custom_gesture_runner.hand_lost(runner_now)
                if fired:
                    try:
                        self.command_detected.emit(f"custom: {fired}")
                    except Exception:
                        pass
                    self._record_action(
                        f"custom:{fired}",
                        f"custom gesture: {fired}",
                    )
            except Exception as exc:
                print(f"[custom-gestures] process error: {exc}")

        if dynamic_runtime is not None and dynamic_runtime.has_dynamic_gestures():
            try:
                swipe_in_flight = self._builtin_open_hand_swipe_in_flight(
                    getattr(result, "prediction", None)
                )
                pred = getattr(result, "prediction", None)
                counting = str(getattr(pred, "stable_label", "") or "") in {"two", "three"}
                loop_custom = False
                try:
                    loop_custom = bool(dynamic_runtime.has_loop_or_complex_templates())
                except Exception:
                    loop_custom = False
                lm, live_hand = self._pick_custom_hand(custom_hands)
                if (
                    (not custom_hands and custom_hands_sampled)
                    or (swipe_in_flight and not loop_custom)
                    or counting
                    or self._pose_sequence_owns_motion(runner_now)
                ):
                    dynamic_runtime.hand_lost()
                elif lm is not None:
                    from ...custom_gestures.dynamic_recording import palm_scale_from_landmarks
                    fired_dyn = dynamic_runtime.process_frame(
                        lm,
                        palm_scale=palm_scale_from_landmarks(lm),
                        handedness=str(live_hand or ""),
                        timestamp=runner_now,
                    )
                    if fired_dyn:
                        try:
                            self.command_detected.emit(f"custom: {fired_dyn}")
                        except Exception:
                            pass
                        self._record_action(
                            f"custom_dynamic:{fired_dyn}",
                            f"custom dynamic gesture: {fired_dyn}",
                        )
            except Exception as exc:
                print(f"[custom-gestures] dynamic runtime error: {exc}")

        if seq_runtime is not None and seq_runtime.has_sequences():
            try:
                lm, live_hand = self._pick_custom_hand(custom_hands)
                if not custom_hands and custom_hands_sampled:
                    seq_runtime.hand_lost(runner_now)
                elif lm is not None:
                    fired_seq = seq_runtime.process_landmarks(
                        lm,
                        handedness=str(live_hand or ""),
                        timestamp=runner_now,
                        # Lone-hand GPU labels often say Left for a
                        # right hand. Never drop a recorded sequence
                        # on that flicker.
                        strict_hand=False,
                        hint_label=str(
                            getattr(getattr(result, "prediction", None), "stable_label", "")
                            or ""
                        ),
                    )
                    if fired_seq:
                        self._seq_suppress_dynamic_until = runner_now + 1.2
                        try:
                            self.command_detected.emit(f"custom: {fired_seq}")
                        except Exception:
                            pass
                        self._record_action(
                            f"custom_sequence:{fired_seq}",
                            f"custom pose sequence: {fired_seq}",
                        )
            except Exception as exc:
                print(f"[custom-gestures] pose sequence runtime error: {exc}")

        # Builtin swipe_left/right fire on the same index-only
        # horizontal motion as a custom "1 swipe right". Preempt
        # the builtin path when a horizontal custom template claims
        # this pose, or for 1.2s after a custom dynamic fire.
        self._suppress_builtin_horizontal_swipe = False
        try:
            if dynamic_runtime is not None and dynamic_runtime.has_dynamic_gestures():
                now_m = time.monotonic()
                if fired_dyn:
                    self._custom_dyn_builtin_suppress_until = now_m + 1.2
                lm = None
                scale = 1.0
                pick_lm, _pick_hand = self._pick_custom_hand(custom_hands)
                if pick_lm is not None:
                    from ...custom_gestures.dynamic_recording import palm_scale_from_landmarks
                    lm = pick_lm
                    scale = float(palm_scale_from_landmarks(pick_lm))
                elif (
                    result.found
                    and result.hand_reading is not None
                    and getattr(result.hand_reading, "landmarks", None) is not None
                ):
                    lm = result.hand_reading.landmarks
                    scale = float(result.hand_reading.palm.scale)
                if (
                    now_m < float(getattr(self, "_custom_dyn_builtin_suppress_until", 0.0) or 0.0)
                    or dynamic_runtime.should_preempt_builtin_horizontal_swipe(
                        lm, palm_scale=scale, now=now_m,
                    )
                ):
                    self._suppress_builtin_horizontal_swipe = True
        except Exception:
            self._suppress_builtin_horizontal_swipe = False

        hand_handedness = result.tracked_hand.handedness if result.found and result.tracked_hand is not None else None
        # MediaPipe occasionally labels a single visible right hand
        # as "Left" when only one hand is in frame and the silhouette
        # could go either way. Two protections layered here:
        #   - When both hands are visible (one Left, one Right),
        #     trust the labels immediately â€” that's unambiguous.
        #   - For single-hand-Left, require ~0.3s of stable Left
        #     labeling before treating it as the user's actual left
        #     hand. A glitchy single-frame misidentification during
        #     a right-hand gesture won't survive 0.3s of consistent
        #     re-labeling, so voice doesn't fire spuriously.
        left_prediction = None
        secondary = getattr(result, "secondary_tracked_hand", None)
        sec_handedness = secondary.handedness if secondary is not None else None
        both_hands_visible = (
            result.found
            and hand_handedness in {"Left", "Right"}
            and sec_handedness in {"Left", "Right"}
            and hand_handedness != sec_handedness
        )
        candidate_left_prediction = None
        candidate_left_reading = None
        if both_hands_visible:
            if hand_handedness == "Left":
                candidate_left_prediction = result.prediction
                candidate_left_reading = result.hand_reading
            else:
                candidate_left_prediction = getattr(result, "secondary_prediction", None)
                candidate_left_reading = getattr(result, "secondary_hand_reading", None)
        elif result.found and hand_handedness == "Left":
            candidate_left_prediction = result.prediction
            candidate_left_reading = result.hand_reading

        left_reading_out = None
        if candidate_left_prediction is not None:
            if self._left_hand_streak_since <= 0.0:
                self._left_hand_streak_since = monotonic_now
            # Two-hand cases are unambiguous so skip the stability
            # gate. Single-hand cases need ~0.3s of stable Left
            # labeling before we act on them.
            if both_hands_visible or (monotonic_now - self._left_hand_streak_since) >= 0.30:
                left_prediction = candidate_left_prediction
                left_reading_out = candidate_left_reading
        else:
            self._left_hand_streak_since = 0.0
        self._left_hand_prediction = left_prediction
        self._left_hand_reading = left_reading_out
        t_gate_a = time.perf_counter() if debug_timing else 0.0
        if self._gestures_enabled:
            if self._drawing_mode_enabled:
                self._volume_mode_active = False
                self._volume_status_text = "paused"
                self._volume_message = "Drawing mode active"
                self._volume_overlay_visible = False
                self._update_volume_overlay()
            else:
                self._handle_volume_control(result, monotonic_now, hand_handedness=hand_handedness)
            t_volume = time.perf_counter() if debug_timing else 0.0
            self._handle_app_controls(result.prediction, result.hand_reading, hand_handedness, monotonic_now)
            t_appctrl = time.perf_counter() if debug_timing else 0.0
        else:
            self._window_pair_pose_metrics(result.hand_reading if hand_handedness == "Right" else None, now=monotonic_now)
            t_volume = time.perf_counter() if debug_timing else 0.0
            t_appctrl = t_volume
        # Pinch grab runs regardless of mode-specific gating above â€”
        # it doesn't compete with mouse / drawing / volume because
        # main_window only forwards its transform when a drawing
        # overlay is actually showing. Cheap when no pinch is held.
        self._handle_pinch_grab(result, monotonic_now)
        self._update_chrome_wheel_overlay(monotonic_now)
        self._update_spotify_wheel_overlay(monotonic_now)
        self._update_youtube_wheel_overlay(monotonic_now)
        self._update_drawing_wheel_overlay(monotonic_now)
        self._update_utility_wheel_overlay(monotonic_now)
        self.voice_status_overlay.tick(monotonic_now)
        self._update_runtime_status()
        t_wheels = time.perf_counter() if debug_timing else 0.0
        # GPU display path: the live-view widgets paint the raw
        # camera frame on the GPU and overlay landmarks via
        # QPainter on top of the texture. We no longer draw
        # landmarks / wheels / mouse monitor / low-fps badge into
        # the BGR frame on the CPU â€” that pipeline was the single
        # biggest CPU cost in the post-engine path (~10-15 ms per
        # frame in heavy modes). `display_frame` is still emitted
        # as a fallback for any consumer that wants the annotated
        # version, but we skip the cv2 mutations to save the cost.
        # Camera-drawing canvas state still has to track frame
        # shape so brush strokes line up â€” that's cheap (no
        # drawing), just a shape lookup.
        display_frame = result.annotated_frame
        self._update_camera_drawing_canvas(display_frame.shape)
        payload = self._build_debug_payload(result, monotonic_now)
        if debug_timing:
            t_draw = time.perf_counter()
        # Wall-clock rate-limit on viewer emits when Lite Mode is on.
        # The receivers can only render ~30 frames/second; at higher
        # emit rates Qt's queued-signal queue grows unbounded and we
        # see massive perceived display lag (the 2-second-delayed
        # camera feeling). Always emit on hand appear/disappear or
        # action-fire so toasts and overlays stay punctual.
        # REMOVED the 30 fps emit throttle that ran in perf modes. It
        # was actively erasing the speedup that Lite/GPU Mode delivers:
        # engine drops from 27ms (normal) to 3-7ms (Lite/GPU), but the
        # emit-side throttle clamped visible fps at 30 either way, so
        # the modes "did nothing" from the user's perspective. The
        # throttle existed to avoid hammering the camera widget at
        # >30 fps but modern displays handle 60-120+ fps fine, and the
        # whole point of perf modes is to give the user MORE fps not
        # less. Now every engine tick emits its display frame. The
        # actual fps cap becomes whichever of:
        #   - engine work time (Lite: ~7ms ceiling 140 fps; GPU: ~6ms
        #     ceiling 160 fps; Normal: ~27ms ceiling 37 fps),
        #   - camera capture rate (YUY2 ~30 fps; ffmpeg-MJPG ~60 fps),
        #   - display refresh / Qt paint coalescing (usually 60 Hz).
        # On a strong PC with Lite or GPU mode + ffmpeg-MJPG camera,
        # actual fps should now reach 50-60. On dad's-PC-class hardware
        # the engine work itself remains the cap; the throttle wasn't
        # helping there either.
        self._last_emit_monotonic = monotonic_now
        try:
            self.debug_frame_ready.emit(display_frame, payload)
        except Exception:
            pass
        if debug_timing:
            t_end = time.perf_counter()
            self._timing_samples.append(
                (
                    t_read - t0,
                    t_prep - t_read,
                    t_engine - t_prep,
                    t_volume - t_gate_a,
                    t_appctrl - t_volume,
                    t_wheels - t_appctrl,
                    t_draw - t_wheels,
                    t_end - t_draw,
                )
            )
            now_secs = time.monotonic()
            if now_secs - self._last_timing_log >= 2.0 and len(self._timing_samples) >= 8:
                samples = list(self._timing_samples)
                self._timing_samples.clear()
                self._last_timing_log = now_secs
                avg_read = sum(s[0] for s in samples) / len(samples) * 1000.0
                avg_prep = sum(s[1] for s in samples) / len(samples) * 1000.0
                avg_engine = sum(s[2] for s in samples) / len(samples) * 1000.0
                avg_vol = sum(s[3] for s in samples) / len(samples) * 1000.0
                avg_app = sum(s[4] for s in samples) / len(samples) * 1000.0
                avg_wheel = sum(s[5] for s in samples) / len(samples) * 1000.0
                avg_overlay = sum(s[6] for s in samples) / len(samples) * 1000.0
                avg_emit = sum(s[7] for s in samples) / len(samples) * 1000.0
                avg_total = (
                    avg_read + avg_prep + avg_engine + avg_vol + avg_app
                    + avg_wheel + avg_overlay + avg_emit
                )
                inferred_fps = 1000.0 / avg_total if avg_total > 0 else 0.0
                try:
                    # Surface mode state in every timing log so you can
                    # see which mode is actually engaged when fps doesn't
                    # match expectations. Useful for the "I toggled Lite
                    # Mode but engine is still 27ms" diagnostic.
                    _mode_lite = bool(getattr(self.config, "lite_mode", False))
                    _mode_gpu = bool(getattr(self.config, "gpu_mode", False))
                    _mode_boost = False
                    _mode_lowfps = self._low_fps_active
                    _mode_tag = (
                        "LOW_FPS" if _mode_lowfps
                        else ("BOOST" if _mode_boost
                              else ("LITE" if _mode_lite
                                    else ("GPU" if _mode_gpu else "NORMAL")))
                    )
                    sys.stderr.write(
                        f"[lite_mode/timing] mode={_mode_tag} "
                        f"(lite={_mode_lite},gpu={_mode_gpu},boost={_mode_boost},"
                        f"lowfps={_mode_lowfps}) "
                        f"read={avg_read:.1f} "
                        f"prep={avg_prep:.1f} engine={avg_engine:.1f} "
                        f"vol={avg_vol:.1f} app={avg_app:.1f} "
                        f"wheel={avg_wheel:.1f} overlay={avg_overlay:.1f} "
                        f"emit={avg_emit:.1f} total={avg_total:.1f}ms "
                        f"(ceiling {inferred_fps:.1f} fps, "
                        f"actual self._fps={self._fps:.1f})\n"
                    )
                    sys.stderr.flush()
                except Exception:
                    pass
        # Loop pacing comes from the periodic 15 ms QTimer. We
        # used to chain singleShot(0, self._tick) here for tighter
        # cadence, but in practice that starved Qt's paint event
        # dispatcher â€” the on-screen update rate collapsed to
        # ~2 fps while the worker kept firing at 30+ fps because
        # the event loop never got enough idle time between ticks
        # for paint events to be processed. The 15 ms periodic
        # timer naturally interleaves paint dispatch and worker
        # execution, which is what keeps the live view smooth.
        # (Tried re-enabling the singleShot on macOS for lower input
        # latency once the clip-cache main-thread grab was gone, but it
        # re-introduced the exact paint-starve: the fps NUMBER rose while
        # the visible feed + landmark bbox lagged behind the actual
        # detection. Reverted — perceived smoothness beats a higher
        # counter. Any future latency work must NOT starve paint dispatch.)

    def _handle_volume_control(self, result, now: float, *, hand_handedness: str | None) -> None:
        # Tutorial isolation: volume + mute go through the
        # volume_tracker pipeline, NOT _dispatch_action — so the
        # tutorial whitelist there doesn't catch them. On any step
        # OTHER than the dedicated volume practice step, reset the
        # tracker and bail so the user's gesture can't crank
        # system volume while they're learning something else. On
        # the volume step itself the pipeline runs normally so the
        # gesture actually changes volume + the engine emits the
        # volume_active / volume_level_scalar / volume_muted fields
        # in the debug payload that the tutorial's volume handler
        # watches for completion.
        if (
            self._tutorial_mode_enabled
            and self._tutorial_step_key != "volume"
        ):
            try:
                self.volume_tracker.reset()
            except Exception:
                pass
            self._volume_mode_active = False
            self._volume_status_text = "tutorial"
            self._volume_overlay_visible = False
            self._volume_dual_active = False
            self._volume_init_palm_x = None
            self._update_volume_overlay()
            return

        if not self.volume_controller.available:
            self._volume_message = self.volume_controller.message
            self._volume_mode_active = False
            self._volume_status_text = "unavailable"
            self._volume_overlay_visible = False
            self._update_volume_overlay()
            return

        # v1.1.7 event-loop optimization (Step 1): only pay the
        # Windows COM cost of reading system volume + mute when there's
        # an actual chance we'll act on it this tick. Fresh state
        # matters when:
        #   - Volume overlay is already visible (user is controlling)
        #   - Volume mode was already active last tick (in-progress)
        #   - The trigger pose is being shown right now (about to enter)
        # Otherwise, use the cached values from the engine's own
        # bookkeeping (_volume_level / _volume_muted) — the volume
        # tracker's update() will no-op with stale values because it's
        # gated on the same pose the fresh-state check is gated on.
        # When the tracker DOES activate on a subsequent tick, the
        # entering_overlay branch below fires refresh_cache() which
        # reads a fresh COM value before the first level adjustment.
        # Savings: ~2-3 ms of main-thread work per idle tick × 25
        # ticks/sec = 50-75 ms/sec of Qt event-loop budget freed up
        # (the 25→35+ fps tick rate improvement the diagnostic release
        # identified as the largest deferred win).
        is_volume_pose_this_tick = (
            hand_handedness == "Right"
            and result.found
            and result.prediction is not None
            and getattr(result.prediction, "stable_label", "") == "volume_pose"
        )
        need_fresh_volume_state = (
            self._volume_overlay_visible
            or self._volume_mode_active
            or is_volume_pose_this_tick
        )
        if need_fresh_volume_state:
            current_level = self.volume_controller.get_level()
            current_muted = self._read_system_mute()
        else:
            current_level = self._volume_level
            current_muted = self._volume_muted
        if self.mouse_tracker.mode_enabled:
            self._volume_level = current_level
            self._volume_muted = current_muted
            self._volume_message = "Mouse mode active"
            self._volume_mode_active = False
            self._volume_status_text = "paused"
            self._volume_overlay_visible = False
            self._volume_dual_active = False
            self._volume_init_palm_x = None
            self._update_volume_overlay()
            return
        if self._drawing_mode_enabled:
            self._volume_level = current_level
            self._volume_muted = current_muted
            self._volume_message = "Drawing mode active"
            self._volume_mode_active = False
            self._volume_status_text = "paused"
            self._volume_overlay_visible = False
            self._volume_dual_active = False
            self._volume_init_palm_x = None
            self._update_volume_overlay()
            return
        if hand_handedness == "Right" and self._effective_dynamic_label(result.prediction) in {"swipe_left", "swipe_right"}:
            self._mute_block_until = max(self._mute_block_until, now + 0.5)

        features = None
        candidate_scores = {}
        landmarks = None
        stable_gesture = "neutral"
        if (
            hand_handedness == "Right"
            and result.found
            and result.tracked_hand is not None
            and result.hand_reading is not None
            and self.engine is not None
        ):
            landmarks = result.tracked_hand.landmarks
            features = self._volume_features_from_hand_reading(result.hand_reading)
            candidate_scores = self.engine.last_static_scores
            stable_gesture = result.prediction.stable_label

        tracker_level = current_level
        if self._volume_dual_active and self._volume_bar_selected == "app" and self._volume_overlay_visible:
            tracker_level = self._volume_app_level if self._volume_app_level is not None else current_level

        palm_roll = None
        if (
            result.found
            and result.tracked_hand is not None
            and result.hand_reading is not None
        ):
            try:
                palm_roll = float(result.hand_reading.palm.roll_deg)
            except Exception:
                palm_roll = None
        update = self.volume_tracker.update(
            features=features,
            landmarks=landmarks,
            candidate_scores=candidate_scores,
            stable_gesture=stable_gesture,
            current_level=tracker_level,
            current_muted=current_muted,
            now=now,
            allow_mute_toggle=now >= self._mute_block_until,
            palm_roll_deg=palm_roll,
        )

        entering_overlay = self._volume_overlay_visible is False and update.overlay_visible
        if entering_overlay:
            refreshed = self.volume_controller.refresh_cache()
            if refreshed.level_scalar is not None:
                current_level = refreshed.level_scalar
            refreshed_muted = self.volume_controller.get_mute()
            if refreshed_muted is not None:
                current_muted = bool(refreshed_muted)
            palm_center = getattr(getattr(result.hand_reading, "palm", None), "center", None) if result.hand_reading is not None else None
            self._volume_init_palm_x = float(palm_center[0]) if palm_center is not None else None
            self._volume_bar_selected = "sys"
            youtube_active = self._youtube_mode_info in {"forced", "auto"}
            if youtube_active:
                app_name, app_level = self.volume_controller.get_app_audio_info(["chrome"])
                if app_name == "chrome":
                    self._volume_dual_active = True
                    self._volume_app_process = "youtube"
                    self._volume_app_label = "YouTube"
                    self._volume_app_level = app_level if app_level is not None else (current_level if current_level is not None else 0.5)
                    self._volume_bar_selected = "app"
                    self.volume_tracker.rebase(self._volume_app_level)
                else:
                    self._volume_dual_active = False
                    self._volume_app_process = ""
                    self._volume_app_label = ""
                    self._volume_app_level = None
            else:
                app_name, app_level = self.volume_controller.get_active_app_audio_info(
                    ["spotify", "chrome"]
                )
                self._volume_dual_active = app_name is not None
                self._volume_app_process = app_name or ""
                self._volume_app_label = app_name.capitalize() if app_name else ""
                if app_name == "spotify":
                    spotify_vol = self.spotify_controller.get_volume()
                    self._volume_app_level = spotify_vol / 100.0 if spotify_vol is not None else (app_level or 0.5)
                else:
                    self._volume_app_level = app_level
            self._volume_app_check_until = now + 3.0

        bar_switched_this_frame = False
        if update.overlay_visible and self._volume_dual_active:
            palm_center = getattr(getattr(result.hand_reading, "palm", None), "center", None) if result.hand_reading is not None else None
            if palm_center is not None and self._volume_init_palm_x is not None:
                dx = float(palm_center[0]) - self._volume_init_palm_x
                previous_bar = self._volume_bar_selected
                if dx < -0.06:
                    self._volume_bar_selected = "app"
                elif dx > 0.06:
                    self._volume_bar_selected = "sys"
                if self._volume_bar_selected != previous_bar:
                    bar_switched_this_frame = True
                    if self._volume_bar_selected == "app":
                        rebase_level = self._volume_app_level if self._volume_app_level is not None else current_level
                    else:
                        rebase_level = current_level
                    self.volume_tracker.rebase(rebase_level)

        controller_error_message: str | None = None
        controller_error_status: str | None = None
        # Tutorial gate: drop the shaka-mute trigger while the
        # tutorial volume step is still in its 'practise up/down
        # first' phase. The tutorial flips _tutorial_block_mute
        # off once swing-up + swing-down are both complete.
        if update.trigger_mute_toggle and getattr(self, "_tutorial_block_mute", False):
            update = update._replace(trigger_mute_toggle=False) if hasattr(update, "_replace") else update
            # If update is a dataclass / plain object that doesn't
            # support _replace, mutate the bool field if it's writable.
            try:
                update.trigger_mute_toggle = False
            except Exception:
                pass
        if update.trigger_mute_toggle:
            toggled = self.volume_controller.toggle_mute()
            if toggled is not None:
                current_muted = toggled
                # Re-seat the 120 ms _read_system_mute cache to the
                # post-toggle value. Without this, the next frame
                # within the cache window reads the STALE pre-toggle
                # value back, then self._volume_muted gets overwritten
                # with the old state, then a frame later the cache
                # expires and re-reads the actual new state -- a
                # spurious False→True→False flicker on the payload's
                # volume_muted field. The tutorial volume step counts
                # those as two transitions, so one shaka press
                # completes the "mute twice" requirement on its own.
                self._mute_cache_value = bool(toggled)
                self._mute_cache_until = time.monotonic() + 0.12
                self.command_detected.emit("Volume mute toggled")
                self._record_action("volume_mute_on" if toggled else "volume_mute_off", "muted" if toggled else "unmuted")
            else:
                controller_error_message = self.volume_controller.message or "mute failed"
                controller_error_status = "error"

        if update.active and update.level is not None and not bar_switched_this_frame:
            if self._volume_dual_active and self._volume_bar_selected == "app":
                new_app = max(0.0, min(1.0, float(update.level)))
                if self._volume_app_process == "spotify":
                    vol_pct = int(round(new_app * 100))
                    self._queue_spotify_volume(vol_pct)
                    self._volume_app_level = new_app
                else:
                    if self.volume_controller.set_app_audio_level(["chrome"], new_app):
                        self._volume_app_level = new_app
                    else:
                        controller_error_message = "app volume adjust failed"
                        controller_error_status = "error"
            else:
                target_level = max(0.0, min(1.0, float(update.level)))
                if self.volume_controller.set_level(target_level):
                    current_level = target_level
                    read_back_level = self.volume_controller.get_level()
                    if read_back_level is not None:
                        current_level = read_back_level
                else:
                    controller_error_message = self.volume_controller.message or "set_level failed"
                    controller_error_status = "error"

        if update.overlay_visible and self._volume_dual_active and now >= self._volume_app_check_until and not (update.active and self._volume_bar_selected == "app"):
            self._volume_app_check_until = now + 3.0
            if self._volume_app_process == "spotify":
                def _refresh_spotify_vol():
                    vol = self.spotify_controller.get_volume()
                    if vol is not None:
                        self._volume_app_level = vol / 100.0
                threading.Thread(target=_refresh_spotify_vol, daemon=True).start()
            else:
                _, fresh_app = self.volume_controller.get_app_audio_info(["chrome"])
                if fresh_app is not None:
                    self._volume_app_level = fresh_app

        if not update.overlay_visible:
            self._volume_dual_active = False
            self._volume_init_palm_x = None

        self._volume_mode_active = update.active
        self._volume_level = current_level if current_level is not None else update.level
        self._volume_muted = current_muted
        self._volume_message = controller_error_message or update.message
        self._volume_status_text = controller_error_status or update.status
        overlay_was_visible = self._volume_overlay_visible
        if entering_overlay:
            self._volume_session_prev_level = current_level
            self._volume_session_prev_app_level = self._volume_app_level
        if overlay_was_visible and not update.overlay_visible:
            self._record_volume_session_end(current_level)
        self._volume_overlay_visible = update.overlay_visible
        self._update_volume_overlay()

    def _record_volume_session_end(self, final_level: float | None) -> None:
        prev = self._volume_session_prev_level
        self._volume_session_prev_level = None
        self._volume_session_prev_app_level = None
        if prev is None or final_level is None:
            return
        delta = float(final_level) - float(prev)
        if abs(delta) < 0.02:
            return
        direction = "up" if delta > 0 else "down"
        pct = int(round(float(final_level) * 100))
        self._record_action(f"volume_{direction}", f"volume {direction} to {pct}%")

    # ----- Gesture Binds dispatch / remap helpers -----------------------
    # The user's per-action pose remap (Settings â†’ Gesture Binds) is
    # applied in two places:
    #   (1) Right-hand prediction at the top of _handle_app_controls.
    #   (2) Left-hand prediction at the top of _handle_left_hand_voice.
    # Both sites pass the prediction through _apply_gesture_binding_remap,
    # which either rewrites stable_label/raw_label so the existing
    # downstream handlers fire the bound static action, OR â€” when the
    # bound action is custom â€” calls _dispatch_action which fires the
    # custom gesture's action via fire_once and returns a neutral
    # prediction so no static handler runs.
    #
    # Custom-side: when the runner detects a custom gesture match, it
    # calls _custom_runner_binding_resolver before fire_once. If the
    # user has remapped that custom gesture to a static action, the
    # resolver dispatches the bound action and returns True, telling
    # the runner to skip its own fire_once.
    # Tutorial → allowed-action whitelist. Each tutorial step
    # exposes ONLY the action it's actively teaching; every other
    # bound action is suppressed in _dispatch_action so the user
    # can't accidentally trigger unrelated functionality (Spotify,
    # mute, dictation, etc.) while learning a specific gesture.
    # Steps with no _dispatch_action-driven action (swipes,
    # gesture_wheel) get an empty set — they're driven by other
    # paths (swipe controller, wheel state machine) that the
    # tutorial code allows directly.
    _TUTORIAL_ALLOWED_ACTIONS: dict[str, frozenset[str]] = {
        "swipes": frozenset(),
        "spotify_open": frozenset({"open_spotify"}),
        "play_pause": frozenset({"play_pause"}),
        "gesture_wheel": frozenset(),
        "mouse_mode": frozenset({"mouse_mode_toggle"}),
        "voice_command": frozenset({"voice_command_listen"}),
    }

    def _tutorial_allowed_actions_for_step(self, step_key: str) -> frozenset[str]:
        return self._TUTORIAL_ALLOWED_ACTIONS.get(step_key, frozenset())

    def _dispatch_action(self, action_id: str, now: float) -> bool:
        """Fire a bound action by id, with a per-action_id cooldown so a
        held pose doesn't spam the action every frame. Returns True if
        the action was dispatched, False if it was suppressed by
        cooldown or could not be performed.

        Side note on "open_gesture_wheel" / "open_screen_wheel": those
        are stateful overlays driven by frame-to-frame hand tracking,
        not single-shot actions. Until the wheel state machines learn
        to honor remapped trigger poses, dispatching them here is a
        no-op so the user gets a clear "nothing happened" rather than
        partial wheel state. The Gesture Binds tab still saves their
        remap; it just won't take effect for those two actions."""
        if not action_id:
            return False
        # Tutorial isolation: while the tutorial window is driving
        # the engine, only the action the current step is teaching
        # is allowed to fire. Any other bound action (e.g., user
        # accidentally does right-hand-two during the swipes step
        # and we'd otherwise launch Spotify in real life) is
        # suppressed so the tutorial flow stays focused. Each step
        # whitelists exactly the action(s) it expects; steps with
        # no static-binding action (swipes, gesture_wheel) get an
        # empty set so EVERYTHING is suppressed.
        if getattr(self, "_tutorial_mode_enabled", False):
            allowed = self._tutorial_allowed_actions_for_step(
                getattr(self, "_tutorial_step_key", "") or ""
            )
            if action_id not in allowed:
                return False
        cooldown_state = getattr(self, "_action_dispatch_last_fire", None)
        if cooldown_state is None:
            cooldown_state = {}
            self._action_dispatch_last_fire = cooldown_state
        # Two-tier gating: a 1.5s "successful fire" window plus a tight
        # 50ms "any attempt" window. The latter prevents disk reads /
        # native API calls from happening every frame while a static
        # pose is held but the action is in cooldown â€” without it,
        # holding a mute-bound-to-custom-action pose for 5 s does up to
        # 150 registry.load() calls and tanks the frame rate.
        last = cooldown_state.get(action_id, 0.0)
        if now - last < 1.5:
            return False
        last_attempt_state = getattr(self, "_action_dispatch_last_attempt", None)
        if last_attempt_state is None:
            last_attempt_state = {}
            self._action_dispatch_last_attempt = last_attempt_state
        last_attempt = last_attempt_state.get(action_id, 0.0)
        if now - last_attempt < 0.05:
            return False
        last_attempt_state[action_id] = now

        fired = False
        try:
            if action_id == "voice_command_listen":
                if not (self._voice_listening or self._dictation_active):
                    self._start_voice_command()
                    fired = True
            # r53: dictation_toggle dispatch removed (dictation temporarily
            # retired). Keeping the dictation code paths alive but unbound
            # from any gesture so voice command / manual triggers still work.
            elif action_id == "mouse_mode_toggle":
                self._mouse_mode_enabled = not self._mouse_mode_enabled
                try:
                    self.voice_status_overlay.show_info_hint(
                        "Mouse Mode: On" if self._mouse_mode_enabled else "Mouse Mode: Off",
                        duration=3.0,
                    )
                except Exception:
                    pass
                fired = True
            elif action_id == "drawing_mode_toggle":
                try:
                    self._toggle_drawing_mode(now)
                    fired = True
                except Exception:
                    fired = False
            elif action_id == "voice_cancel":
                if self._voice_listening or self._dictation_active or self._save_prompt_active or self._selection_prompt_active:
                    try:
                        self._cancel_all_voice_stages()
                        fired = True
                    except Exception:
                        fired = False
            elif action_id == "open_spotify":
                try:
                    # Tutorial spotify_open step: launch Spotify in the
                    # BACKGROUND so it doesn't steal focus from the
                    # tutorial window. The user just needs the engine
                    # to confirm "yes, Spotify opened" — we don't
                    # need to bring it to the foreground while the
                    # tutorial walks them through the next step.
                    in_tutorial_spotify = (
                        self._tutorial_mode_enabled
                        and self._tutorial_step_key == "spotify_open"
                    )
                    if in_tutorial_spotify:
                        # Only spawn the launcher when there's no
                        # real Spotify process yet — otherwise leave
                        # the existing window alone (don't focus).
                        if not self.spotify_controller._has_real_spotify_process():
                            threading.Thread(
                                target=lambda: self.spotify_controller.launch_spotify(hidden=True),
                                name="spotify-tutorial-bg",
                                daemon=True,
                            ).start()
                        fired = True
                    else:
                        # EnumWindows + process walk + possible launch
                        # used to run on the gesture/UI thread and hitch
                        # the live view for a split second. Same work,
                        # background thread.
                        threading.Thread(
                            target=self.spotify_controller.focus_or_open_window,
                            name="spotify-focus",
                            daemon=True,
                        ).start()
                        fired = True
                except Exception:
                    fired = False
            elif action_id == "play_pause":
                try:
                    if sys.platform == "darwin":
                        # macOS: control the Spotify app directly via AppleScript
                        # when it's running (reliable, and matches the Windows
                        # fist, which is Spotify-specific). Fall back to the OS
                        # media key for other players when Spotify isn't running.
                        if self.spotify_controller.is_running():
                            self.spotify_controller.dispatch_async(self.spotify_controller.toggle_playback)
                            fired = True
                        else:
                            from ...platform_compat import mac_input
                            fired = mac_input.media_play_pause()
                    else:
                        self.spotify_controller.dispatch_async(self.spotify_controller.toggle_playback)
                        fired = True
                except Exception:
                    fired = False
            elif action_id == "open_chrome":
                try:
                    threading.Thread(
                        target=self.chrome_controller.focus_or_open_window,
                        name="chrome-focus",
                        daemon=True,
                    ).start()
                    fired = True
                except Exception:
                    fired = False
            elif action_id == "open_touchless":
                # Main window connects to open_touchless_requested and
                # raises itself. Engine has no direct main-window ref,
                # so the signal hop is necessary.
                try:
                    self.open_touchless_requested.emit()
                    fired = True
                except Exception:
                    fired = False
            elif action_id == "instant_clip":
                # Fire-and-forget: emit signal, main window does the
                # actual export on a worker thread. No prompts, no
                # save dialog — just save to clips_save_dir.
                try:
                    self.instant_clip_requested.emit()
                    fired = True
                    try:
                        self.command_detected.emit("Instant clip saved")
                    except Exception:
                        pass
                except Exception:
                    fired = False
            elif action_id == "system_mute_toggle":
                try:
                    toggled = self.volume_controller.toggle_mute()
                    fired = toggled is not None
                    if fired:
                        # Match the volume-pose path: re-seat the
                        # _read_system_mute cache to the post-toggle
                        # value so the next frame within the 120 ms
                        # cache window doesn't return the stale
                        # pre-toggle state.
                        self._mute_cache_value = bool(toggled)
                        self._mute_cache_until = time.monotonic() + 0.12
                        try:
                            self.command_detected.emit("Volume mute toggled")
                        except Exception:
                            pass
                except Exception:
                    fired = False
            elif action_id in ("open_gesture_wheel", "open_screen_wheel"):
                # Wheel overlays are stateful. Skipping for now â€” see
                # docstring above.
                fired = False
            elif action_id.startswith("custom_action:"):
                name = action_id.split(":", 1)[1]
                # Reuse the runner's already-loaded registry instead of
                # opening + parsing the JSON file on every dispatch.
                # Falls back to a fresh load only when the runner isn't
                # available (shouldn't happen in practice â€” the runner
                # is constructed at engine init).
                try:
                    from ...custom_gestures.action import fire_once
                    runner = getattr(self, "_custom_gesture_runner", None)
                    registry = getattr(runner, "_registry", None) if runner is not None else None
                    if registry is None:
                        from ...custom_gestures.registry import GestureRegistry
                        registry = GestureRegistry()
                        registry.load()
                    gesture = registry.get(name)
                    if gesture is not None:
                        fired = bool(fire_once(gesture.name, gesture.action))
                except Exception:
                    fired = False
        except Exception:
            fired = False

        if fired:
            cooldown_state[action_id] = now
            # Most fire paths (volume, swipes, wheels, voice,
            # drawing) emit action_fired via _record_action, the
            # universal action-history hook. A handful of
            # _dispatch_action branches (mouse_mode_toggle,
            # open_spotify, play_pause, system_mute_toggle) skip
            # _record_action so we cover them here, with a small
            # dedupe window so cases that DO go through both paths
            # (voice_command_listen, dictation_toggle,
            # drawing_mode_toggle) only emit once.
            try:
                last_telemetry = getattr(self, "_action_fired_telemetry_last", None)
                if last_telemetry is None:
                    last_telemetry = {}
                    self._action_fired_telemetry_last = last_telemetry
                if now - last_telemetry.get(action_id, 0.0) >= 0.25:
                    last_telemetry[action_id] = now
                    from ...telemetry import track as _track
                    _track(
                        "action_fired",
                        {
                            "action_id": str(action_id),
                            "in_tutorial": bool(getattr(self, "_tutorial_mode_enabled", False)),
                        },
                    )
            except Exception:
                pass
        return fired

    def _maybe_fire_open_chrome_touchless(
        self,
        prediction,
        hand_handedness: str | None,
        now: float,
    ) -> None:
        """Right-hand plain-three -> open_chrome,
        right-hand plain-four -> open_touchless. Fire after a 1.0 s
        hold with a 2.0 s cooldown so a quick gesture doesn't double-
        fire when the user is in transit between poses.

        Also inspects the SECONDARY hand: if the user's right hand
        appears in frame as the secondary while their left is primary
        (rare but happens when two hands are tracked and the left was
        detected first), we still fire on the right. Without this,
        users who keep both hands in view get inconsistent triggering.

        Only fires for the DEFAULT pose binding. If the user has
        remapped right_three or right_four to a different action, the
        binding remap step above this call already rewrote the
        stable_label, so the labels here will be different and this
        method early-exits. No competing path."""
        seq = getattr(self, "_pose_sequence_runtime", None)
        if seq is not None and seq.has_sequences() and seq.is_in_progress():
            # Mid-count 3→2→1 owns the right-hand "three"; do not
            # also arm open-Chrome on that hold.
            return
        # Resolve which prediction is from the RIGHT hand. _handle_app_controls
        # is called with the primary prediction + its handedness. If the
        # primary is the right hand, use it. Otherwise, walk the engine's
        # cached right-hand state for the secondary's prediction.
        if hand_handedness == "Right":
            right_pred = prediction
        else:
            # _normalize_result_right_primary already moved the right hand
            # to primary when it was present, so reaching here means the
            # right hand isn't tracked this frame. Nothing to fire.
            if self._open_action_candidate is not None:
                self._open_action_candidate = None
                self._open_action_candidate_since = 0.0
            return
        stable_label = str(getattr(right_pred, "stable_label", "neutral") or "neutral")
        # Rate-limited visibility log: every ~1 s, surface what the
        # right-hand recognizer is producing. Lets the user diagnose
        # "I'm making the gesture but nothing fires" by reading the
        # debug log — they can see whether the engine sees 'four',
        # 'four_together', 'neutral', or something else entirely.
        if (
            stable_label != self._open_action_last_diag_label
            or (now - self._open_action_last_diag_at) > 1.0
        ):
            self._open_action_last_diag_label = stable_label
            self._open_action_last_diag_at = now
            try:
                self.engine_log.emit(f"right-hand label: {stable_label}")
            except Exception:
                pass
        # Only plain three / plain four trigger this — the _together
        # variants are handled by their bound mode toggles.
        if stable_label not in {"three", "four"}:
            if self._open_action_candidate is not None:
                self._open_action_candidate = None
                self._open_action_candidate_since = 0.0
            return
        if self._open_action_candidate != stable_label:
            self._open_action_candidate = stable_label
            self._open_action_candidate_since = now
            try:
                self.engine_log.emit(
                    f"open-action: candidate started '{stable_label}' on right hand"
                )
            except Exception:
                pass
            return
        if now < self._open_action_cooldown_until:
            return
        held = now - self._open_action_candidate_since
        if held < _STATIC_GESTURE_HOLD_SECONDS:
            return
        self._open_action_cooldown_until = now + 2.0
        action_id = "open_chrome" if stable_label == "three" else "open_touchless"
        try:
            self.engine_log.emit(
                f"open-action: firing {action_id} after {held:.2f}s hold"
            )
        except Exception:
            pass
        try:
            self._dispatch_action(action_id, now)
        except Exception:
            pass

    def _apply_gesture_binding_remap(self, prediction, hand_handedness, now: float):
        """Apply the user's static-pose binding remap to a prediction.

        Returns the (possibly rewritten) prediction. When the bound
        action is custom, also fires the custom gesture's action via
        _dispatch_action and returns a prediction with stable_label /
        raw_label set to neutral so the static handlers don't double-
        process the gesture.

        No-op if the user has not remapped the action away from its
        default pose, or if the prediction's label doesn't correspond
        to a known static pose. Cross-handedness remaps are skipped
        (the engine's left/right code paths are too divergent for a
        single-frame label rewrite to be safe)."""
        if prediction is None or not hand_handedness:
            return prediction
        stable = str(getattr(prediction, "stable_label", "neutral") or "neutral")
        if stable == "neutral":
            return prediction
        pose_id = pose_id_for_static_label(hand_handedness, stable)
        if pose_id is None:
            return prediction
        cfg = self.config
        bound_action_id = action_bound_to_pose(cfg, pose_id)
        if bound_action_id is None:
            return prediction
        # Default mapping: nothing to do.
        if default_pose_for_action(bound_action_id) == pose_id:
            return prediction

        if bound_action_id.startswith("custom_action:"):
            # Static pose -> custom action. Fire the custom gesture's
            # stored action and neutralize the prediction so the static
            # handlers don't run.
            self._dispatch_action(bound_action_id, now)
            return self._neutralize_prediction(prediction)

        # Static pose -> static action. Rewrite the prediction's labels
        # to the bound action's default pose so the existing engine
        # code paths fire the right thing.
        target_pose = default_pose_for_action(bound_action_id)
        target_label = static_label_for_pose_id(target_pose) if target_pose else None
        if target_label is None:
            return prediction
        target_handedness, target_raw_label = target_label
        if target_handedness != hand_handedness:
            # Cross-handedness â€” skip remap (see docstring).
            return prediction
        return self._rewrite_prediction_labels(prediction, target_raw_label)

    def _effective_dynamic_label(self, prediction) -> str:
        """Builtin swipe_left/right, with custom-horizontal preemption.

        When a custom index-only swipe-right owns the pose, the live
        app must not also dispatch the builtin swipe. Returns
        "neutral" in that case; otherwise the prediction's label.
        """
        label = str(getattr(prediction, "dynamic_label", "neutral") or "neutral")
        if (
            getattr(self, "_suppress_builtin_horizontal_swipe", False)
            and label in {"swipe_left", "swipe_right"}
        ):
            return "neutral"
        if label == "repeat_circle":
            stable = str(getattr(prediction, "stable_label", "") or "")
            if stable in {"two", "three", "four"}:
                return "neutral"
            if self._pose_sequence_owns_motion():
                return "neutral"
        return label

    def _pose_sequence_owns_motion(self, now: float | None = None) -> bool:
        """True while a 3→2→1 (or similar) sequence is mid-count, or
        briefly after it fires, so repeat_circle / custom circle cannot
        steal the last 'one' pose."""
        seq = getattr(self, "_pose_sequence_runtime", None)
        if seq is not None and seq.has_sequences() and seq.is_in_progress():
            return True
        until = float(getattr(self, "_seq_suppress_dynamic_until", 0.0) or 0.0)
        if until <= 0.0:
            return False
        now_m = float(now if now is not None else time.monotonic())
        return now_m < until

    @staticmethod
    def _neutralize_prediction(prediction):
        """Return a copy of `prediction` with stable_label and raw_label
        set to "neutral". Uses dataclasses.replace when available so
        the GesturePrediction dataclass keeps the same identity; falls
        back to a SimpleNamespace clone for non-dataclass shapes."""
        try:
            from dataclasses import is_dataclass, replace
            if is_dataclass(prediction):
                return replace(prediction, stable_label="neutral", raw_label="neutral")
        except Exception:
            pass
        try:
            from types import SimpleNamespace
            attrs = {k: getattr(prediction, k) for k in dir(prediction) if not k.startswith("_")}
            attrs["stable_label"] = "neutral"
            attrs["raw_label"] = "neutral"
            return SimpleNamespace(**attrs)
        except Exception:
            return prediction

    @staticmethod
    def _rewrite_prediction_labels(prediction, new_label: str):
        """Return a copy of `prediction` with stable_label / raw_label
        rewritten to `new_label`. Same dataclass-aware fallback as
        _neutralize_prediction."""
        try:
            from dataclasses import is_dataclass, replace
            if is_dataclass(prediction):
                return replace(prediction, stable_label=new_label, raw_label=new_label)
        except Exception:
            pass
        try:
            from types import SimpleNamespace
            attrs = {k: getattr(prediction, k) for k in dir(prediction) if not k.startswith("_")}
            attrs["stable_label"] = new_label
            attrs["raw_label"] = new_label
            return SimpleNamespace(**attrs)
        except Exception:
            return prediction

    def _custom_runner_can_use_engine_landmarks(self) -> bool:
        """Whether the engine's hand-tracking output is close enough to
        the recorder's MediaPipe landmarks that the runner can reuse
        them and skip its own private MediaPipe pass.

        Three live backends today:
          - `mediapipe-cpu`: MediaPipe Hands on CPU, model_complexity=1
            by default â€” identical landmarks to the recorder. Safe to
            reuse.
          - `mediapipe-tasks-gpu`: MediaPipe Tasks API HandLandmarker on
            GPU. Same models as solutions.hands per the runtime
            docstring, so coordinates match within a pixel. Safe to
            reuse.
          - `onnx-directml`: OpenCV Zoo ONNX export of the same
            MediaPipe Hands weights run on DirectML. Per the
            onnx_runtime.py docstring: 'same weights as
            mediapipe.solutions.hands so accuracy is identical'. Pre-
            and post-processing differ slightly so coordinates can
            drift a couple of pixels, but the alternative is paying
            5â€“10 ms/frame on a private MediaPipe CPU pass on the main
            thread â€” which on phone-camera setups is exactly what
            drops the live FPS to 14â€“22. Reusing engine landmarks here
            is a clear win for fluidity; if a user notices a custom
            gesture missing in GPU mode they can re-record it from the
            same mode.

        Lite mode (model_complexity=0) is excluded regardless of
        backend â€” the lite landmark model produces visibly different
        coordinates that drop classifier scores ~0.05â€“0.10 below the
        recorder's baseline.

        Darwin is the exception: a second MediaPipe pass for custom
        gestures costs 15-25 ms/frame on Apple Silicon and collapses
        Lite / GPU / Boost to the same ~12 fps as Default whenever a
        custom gesture exists (even one that never matches). Reuse
        live-engine landmarks there; scores may sit a few points lower
        than a complexity=1 recording."""
        if sys.platform == "darwin":
            return True
        try:
            engine = getattr(self, "engine", None)
            detector = getattr(engine, "detector", None) if engine is not None else None
            runtime = getattr(detector, "runtime", None) if detector is not None else None
            backend = getattr(runtime, "backend", "") if runtime is not None else ""
            model_complexity = int(getattr(detector, "model_complexity", 1)) if detector is not None else 1
        except Exception:
            return False
        if backend not in ("mediapipe-cpu", "mediapipe-tasks-gpu"):
            return False
        if model_complexity != 1:
            return False
        return True

    def _custom_hands_this_frame(self, result, frame) -> tuple[list, bool]:
        """Landmarks for custom static / dynamic / pose-sequence.

        Returns (hands, sampled). `sampled` is True on a fresh engine
        or MediaPipe observation. False means the MediaPipe path was
        rate-limited and `hands` is the previous sample — callers must
        not treat an empty list as hand_lost in that case.

        Recorders always use MediaPipe Hands at complexity=1. GPU
        (ONNX) and Lite landmarks are a different distribution, so
        those modes run a private MediaPipe pass on the detection
        frame (already selfie-mirrored) rather than engine output.
        When the engine runner attached a payload on its thread,
        reuse that so this method stays cheap on the GUI thread.
        """
        attached = getattr(result, "custom_hands_payload", None)
        if attached is not None:
            hands, sampled = attached
            return (list(hands) if hands else []), bool(sampled)
        return GestureWorker._compute_custom_hands(self, result, frame)

    def _attach_custom_hands_on_engine_thread(self, frame, result) -> None:
        """Run GPU/Lite custom MediaPipe on the engine thread.

        Called after process_frame, before the queued GUI delivery.
        Stashes (hands, sampled) on the result so the GUI path does
        not pay 5–10 ms of MediaPipe on the camera tick.
        """
        if result is None:
            return
        need_custom = False
        try:
            if self._custom_gesture_runner is not None and self._custom_gesture_runner.has_gestures:
                need_custom = True
            dynamic_runtime = getattr(self, "_dynamic_gesture_runtime", None)
            if dynamic_runtime is not None and dynamic_runtime.has_dynamic_gestures():
                need_custom = True
            seq_runtime = getattr(self, "_pose_sequence_runtime", None)
            if seq_runtime is not None and seq_runtime.has_sequences():
                need_custom = True
        except Exception:
            need_custom = bool(self._custom_gesture_runner is not None)
        if not need_custom:
            result.custom_hands_payload = ([], True)
            return
        try:
            result.custom_hands_payload = GestureWorker._compute_custom_hands(
                self, result, frame
            )
        except Exception:
            result.custom_hands_payload = ([], True)

    def _compute_custom_hands(self, result, frame) -> tuple[list, bool]:
        """Landmarks for custom static / dynamic / pose-sequence.

        Returns (hands, sampled). `sampled` is True on a fresh engine
        or MediaPipe observation. False means the MediaPipe path was
        rate-limited and `hands` is the previous sample — callers must
        not treat an empty list as hand_lost in that case.

        Recorders always use MediaPipe Hands at complexity=1. GPU
        (ONNX) and Lite landmarks are a different distribution, so
        those modes run a private MediaPipe pass on the detection
        frame (already selfie-mirrored) rather than engine output.
        """
        if self._custom_runner_can_use_engine_landmarks():
            return self._build_engine_hands_for_runner(result), True
        self._custom_runner_slow_path_counter = (
            (self._custom_runner_slow_path_counter + 1)
            % max(1, self._custom_runner_slow_path_skip_ratio)
        )
        should_run = self._custom_runner_slow_path_counter == 0
        if not should_run:
            cached = getattr(self, "_last_custom_mp_hands", None)
            return (list(cached) if cached else []), False
        frame_for_mp = frame if frame is not None else getattr(result, "annotated_frame", None)
        runner = getattr(self, "_custom_gesture_runner", None)
        if runner is None or frame_for_mp is None:
            self._last_custom_mp_hands = []
            return [], True
        try:
            hands = runner.extract_hands(frame_for_mp)
        except Exception:
            hands = []
        self._last_custom_mp_hands = list(hands) if hands else []
        return self._last_custom_mp_hands, True

    @staticmethod
    def _pick_custom_hand(hands: list):
        """First (landmarks, handedness) pair, or (None, None)."""
        if not hands:
            return None, None
        lm, label = hands[0]
        return lm, label

    @staticmethod
    def _build_engine_hands_for_runner(result) -> list:
        """Collect (landmarks, handedness) tuples from the engine's
        per-frame result so the runner can classify without running its
        own MediaPipe pass. Returns an empty list when no hands were
        detected, in which case the caller should send hand_lost()
        instead."""
        out: list = []
        primary = getattr(result, "tracked_hand", None)
        if primary is not None and getattr(primary, "landmarks", None) is not None:
            out.append((
                primary.landmarks,
                getattr(primary, "handedness", None) or None,
            ))
        secondary = getattr(result, "secondary_tracked_hand", None)
        if secondary is not None and getattr(secondary, "landmarks", None) is not None:
            out.append((
                secondary.landmarks,
                getattr(secondary, "handedness", None) or None,
            ))
        return out

    def _custom_runner_binding_resolver(self, gesture_name: str) -> bool:
        """Called by CustomGestureRunner just before fire_once on the
        firing edge. If the user has remapped this custom gesture to a
        non-default action, dispatch that action and return True so the
        runner skips its own fire. Return False to let the runner fire
        the gesture's stored action (default behavior)."""
        if not gesture_name:
            return False
        pose_id = f"custom:{gesture_name}"
        try:
            bound_action_id = action_bound_to_pose(self.config, pose_id)
        except Exception:
            return False
        if bound_action_id is None:
            return False
        if bound_action_id == f"custom_action:{gesture_name}":
            return False  # default â€” runner fires stored action
        try:
            return bool(self._dispatch_action(bound_action_id, time.monotonic()))
        except Exception:
            return False

    def _custom_runner_image_overlay_handler(self, filename: str) -> None:
        """Bridge from the custom-gesture runner (worker thread) to
        the main window (GUI thread). Just emits a Qt signal â€” the
        receiver runs on the GUI thread because Qt picks a queued
        connection across threads â€” and main_window does the actual
        path resolution and overlay-widget mutation there.
        Logs to recent-actions so the user can see the fire in the
        same activity feed as built-in gestures."""
        try:
            self.drawing_overlay_toggle_requested.emit(str(filename or ""))
        except Exception:
            pass
        try:
            self._record_action(
                "drawing_overlay_toggle",
                f"toggled drawing overlay: {filename}" if filename else "toggled drawing overlay",
            )
        except Exception:
            pass

    def reset_pinch_grab_state(self) -> None:
        """Clear cumulative transform + active mode. Called by
        main_window when the drawing overlay is hidden (toggled off
        or swapped to a different drawing) so the next pinch starts
        from a clean baseline instead of inheriting the previous
        drawing's offset."""
        self._pinch_mode = "none"
        self._pinch_one_slot = None
        self._pinch_anchor_palm = None
        self._pinch_two_anchor_dist = 0.0
        self._pinch_two_anchor_mid = (0.0, 0.0)
        self._pinch_accum_dx = 0.0
        self._pinch_accum_dy = 0.0
        self._pinch_accum_scale = 1.0
        self._pinch_last_seen_primary = 0.0
        self._pinch_last_seen_secondary = 0.0
        self._pinch_streak_start_primary = 0.0
        self._pinch_streak_start_secondary = 0.0
        self._pinch_smoothed_primary = None
        self._pinch_smoothed_secondary = None

    def _smooth_palm(self, slot: str, raw: Optional[tuple[float, float]]) -> Optional[tuple[float, float]]:
        """Velocity-adaptive smoother for palm coords used by the
        pinch-grab math. The fixed-alpha EMA we shipped first felt
        bouncy on foreshortened pinches because MediaPipe's
        per-frame landmark jitter is 3-10 px even when the user is
        holding still â€” a fixed alpha had to choose between 'too
        laggy on real motion' and 'too jittery when held' and both
        were bad. Adaptive alpha solves it: tiny per-frame deltas
        (= jitter while holding) get heavy damping; medium deltas
        pass through with moderate damping; large deltas (= the
        user actually moving) react fast.

        Also clamps single-frame jumps > 10% of normalised screen
        width to that 10% maximum â€” that catches the case where
        MediaPipe completely loses the hand and re-detects it at
        a far-away position, which would otherwise teleport the
        grabbed drawing on the next emit.
        """
        if raw is None:
            return getattr(self, f"_pinch_smoothed_{slot}", None)
        prev = getattr(self, f"_pinch_smoothed_{slot}", None)
        if prev is None:
            new = raw
        else:
            dx = raw[0] - prev[0]
            dy = raw[1] - prev[1]
            dist = (dx * dx + dy * dy) ** 0.5
            # Velocity-clamp: cap the maximum delta we'll consider
            # before applying alpha. A genuine fast hand move is
            # ~5-10% screen / frame; >10% almost always means a
            # tracker re-detection jump.
            if dist > 0.10 and dist > 1e-6:
                clamp_scale = 0.10 / dist
                target_x = prev[0] + dx * clamp_scale
                target_y = prev[1] + dy * clamp_scale
            else:
                target_x = raw[0]
                target_y = raw[1]
            # Adaptive alpha. Tiers calibrated against typical
            # MediaPipe foreshortened-pinch jitter (~0.005-0.015
            # normalised units) vs. genuine slow / fast hand
            # motion. Holding still â†’ alpha 0.10 (smoother barely
            # moves, killing jitter). Real motion â†’ alpha 0.55
            # (snaps to current position).
            if dist < 0.015:
                alpha = 0.10
            elif dist < 0.05:
                alpha = 0.25
            else:
                alpha = 0.55
            new = (
                alpha * target_x + (1.0 - alpha) * prev[0],
                alpha * target_y + (1.0 - alpha) * prev[1],
            )
        setattr(self, f"_pinch_smoothed_{slot}", new)
        return new

    def _palm_xy(self, hand_reading) -> Optional[tuple[float, float]]:
        """Pull (x, y) out of a HandReading's palm.center. Returns
        None for the no-hand or malformed-reading case so callers
        can short-circuit cleanly."""
        if hand_reading is None:
            return None
        try:
            center = hand_reading.palm.center
            return (float(center[0]), float(center[1]))
        except Exception:
            return None

    def _handle_pinch_grab(self, result, now: float) -> None:
        """Pinch-to-grab + bimanual-stretch driver.

        Called every frame after the engine result lands. Walks the
        primary + secondary hand predictions to decide which mode
        is active (none / one-hand / two-hand) and emits an
        absolute (cumulative-since-overlay-shown) transform when a
        pinch is held. Stays silent when no overlay is visible
        anyway â€” main_window only forwards the signal when
        DrawingOverlayWindow is showing.

        Includes a sticky-active grace window so brief recogniser
        flicker (foreshortened pinch landmarks bouncing between
        labels for a frame or two) doesn't drop the grab and force
        an anchor reset; and EMA-smooths the palm positions used
        for the actual transform math so jittery landmarks don't
        translate into a jittery on-screen drawing."""
        primary_pred = getattr(result, "prediction", None)
        secondary_pred = getattr(result, "secondary_prediction", None)
        primary_pinch = (
            primary_pred is not None
            and getattr(primary_pred, "stable_label", "neutral") == "pinch"
        )
        secondary_pinch = (
            secondary_pred is not None
            and getattr(secondary_pred, "stable_label", "neutral") == "pinch"
        )
        primary_palm = self._palm_xy(getattr(result, "hand_reading", None))
        secondary_palm = self._palm_xy(getattr(result, "secondary_hand_reading", None))

        # Sticky-active grace + pre-activation streak tracking.
        #
        # last_seen: monotonic time of the most recent frame where
        #   the recogniser actually labelled this slot 'pinch'.
        # streak_start: when the CURRENT continuous pinch streak
        #   began. Continuous = label seen within grace; an
        #   uninterrupted >= grace_seconds gap clears the streak.
        # The streak only counts as 'active' once it has lasted
        # _pinch_activation_delay seconds â€” that's the pre-roll
        # the user asked for so the drawing doesn't jump the
        # instant a transient pinch label stabilises.
        if primary_pinch:
            in_primary_grace = (now - self._pinch_last_seen_primary) < self._pinch_grace_seconds
            if not in_primary_grace or self._pinch_streak_start_primary <= 0.0:
                self._pinch_streak_start_primary = now
            self._pinch_last_seen_primary = now
        elif (now - self._pinch_last_seen_primary) >= self._pinch_grace_seconds:
            self._pinch_streak_start_primary = 0.0

        if secondary_pinch:
            in_secondary_grace = (now - self._pinch_last_seen_secondary) < self._pinch_grace_seconds
            if not in_secondary_grace or self._pinch_streak_start_secondary <= 0.0:
                self._pinch_streak_start_secondary = now
            self._pinch_last_seen_secondary = now
        elif (now - self._pinch_last_seen_secondary) >= self._pinch_grace_seconds:
            self._pinch_streak_start_secondary = 0.0

        primary_in_grace = primary_pinch or (
            (now - self._pinch_last_seen_primary) < self._pinch_grace_seconds
        )
        secondary_in_grace = secondary_pinch or (
            (now - self._pinch_last_seen_secondary) < self._pinch_grace_seconds
        )
        primary_held_long_enough = (
            self._pinch_streak_start_primary > 0.0
            and (now - self._pinch_streak_start_primary) >= self._pinch_activation_delay
        )
        secondary_held_long_enough = (
            self._pinch_streak_start_secondary > 0.0
            and (now - self._pinch_streak_start_secondary) >= self._pinch_activation_delay
        )
        primary_active = primary_in_grace and primary_held_long_enough
        secondary_active = secondary_in_grace and secondary_held_long_enough

        # Mode + slot decision. Two-hand stretch needs both slots
        # active; otherwise EITHER slot pinching alone is a
        # one-hand grab â€” works with the user's left, right, or
        # whichever single hand is in frame after the right-as-
        # primary normalisation.
        if primary_active and secondary_active and primary_palm and secondary_palm:
            new_mode = "two"
            new_one_slot: Optional[str] = None
        elif primary_active and primary_palm:
            new_mode = "one"
            new_one_slot = "primary"
        elif secondary_active and secondary_palm:
            new_mode = "one"
            new_one_slot = "secondary"
        else:
            new_mode = "none"
            new_one_slot = None

        # ALWAYS run the smoother when there's a raw reading,
        # regardless of the mode we're transitioning to. The
        # previous version nulled primary_palm_s / secondary_palm_s
        # the moment new_mode == 'none' â€” but that ran BEFORE the
        # lock-in below, so the release frame's smoothed palm was
        # gone before the lock-in could read it. Result: the
        # cumulative offset never absorbed the user's drag, and
        # the next pinch session started from the OLD accum,
        # snapping the drawing back to its pre-grab position.
        # Now the smoother stays valid through the transition;
        # the post-transition reset (further down) wipes it only
        # AFTER lock-in has consumed the smoothed values.
        primary_palm_s = self._smooth_palm("primary", primary_palm)
        secondary_palm_s = self._smooth_palm("secondary", secondary_palm)

        def _palm_for_slot(slot: Optional[str]) -> Optional[tuple[float, float]]:
            if slot == "primary":
                return primary_palm_s
            if slot == "secondary":
                return secondary_palm_s
            return None

        # --- mode transitions. Trigger on string-mode change OR
        # one-hand slot change (leftâ†”right swap mid-grab). Lock
        # the live delta of the OUTGOING configuration into the
        # cumulative state, then capture anchors for the INCOMING
        # configuration. Uses smoothed palm coords throughout so
        # the lock-in matches what the user actually saw drawn.
        slot_changed = (
            new_mode == "one"
            and self._pinch_mode == "one"
            and new_one_slot != self._pinch_one_slot
        )
        if new_mode != self._pinch_mode or slot_changed:
            outgoing_palm = _palm_for_slot(self._pinch_one_slot)
            if (
                self._pinch_mode == "one"
                and self._pinch_anchor_palm is not None
                and outgoing_palm is not None
            ):
                self._pinch_accum_dx += outgoing_palm[0] - self._pinch_anchor_palm[0]
                self._pinch_accum_dy += outgoing_palm[1] - self._pinch_anchor_palm[1]
            elif (
                self._pinch_mode == "two"
                and primary_palm_s is not None
                and secondary_palm_s is not None
            ):
                cur_mid = (
                    (primary_palm_s[0] + secondary_palm_s[0]) * 0.5,
                    (primary_palm_s[1] + secondary_palm_s[1]) * 0.5,
                )
                cur_dist = max(
                    1e-4,
                    ((primary_palm_s[0] - secondary_palm_s[0]) ** 2
                     + (primary_palm_s[1] - secondary_palm_s[1]) ** 2) ** 0.5,
                )
                self._pinch_accum_dx += cur_mid[0] - self._pinch_two_anchor_mid[0]
                self._pinch_accum_dy += cur_mid[1] - self._pinch_two_anchor_mid[1]
                if self._pinch_two_anchor_dist > 1e-4:
                    raw_ratio = cur_dist / self._pinch_two_anchor_dist
                    # Sensitivity gain: amplify scale changes around
                    # 1.0 so a small palm-distance change produces
                    # a more decisive stretch / squish. The
                    # palm-position smoother heavily damps slow
                    # motion (alpha 0.10 below 1.5% screen / frame)
                    # to kill jitter â€” that damping was making slow
                    # stretches feel sluggish. Amplifying the scale
                    # delta here recovers responsiveness without
                    # weakening the smoother. Linear amplification
                    # is symmetric: same gain applied to expand
                    # (>1) and squish (<1).
                    self._pinch_accum_scale *= 1.0 + _PINCH_SCALE_SENSITIVITY * (raw_ratio - 1.0)
            # Capture incoming-mode anchors.
            incoming_palm = _palm_for_slot(new_one_slot)
            if new_mode == "one" and incoming_palm is not None:
                self._pinch_anchor_palm = incoming_palm
            elif new_mode == "two" and primary_palm_s is not None and secondary_palm_s is not None:
                self._pinch_two_anchor_mid = (
                    (primary_palm_s[0] + secondary_palm_s[0]) * 0.5,
                    (primary_palm_s[1] + secondary_palm_s[1]) * 0.5,
                )
                self._pinch_two_anchor_dist = max(
                    1e-4,
                    ((primary_palm_s[0] - secondary_palm_s[0]) ** 2
                     + (primary_palm_s[1] - secondary_palm_s[1]) ** 2) ** 0.5,
                )
            else:
                self._pinch_anchor_palm = None
            # Notify subscribers about active-state edge changes.
            was_active = self._pinch_mode != "none"
            now_active = new_mode != "none"
            self._pinch_mode = new_mode
            self._pinch_one_slot = new_one_slot
            if was_active != now_active:
                try:
                    self.drawing_overlay_grab_active.emit(now_active)
                except Exception:
                    pass

        # --- live transform every frame the mode is active. Same
        # smoothed-coord story: the displayed transform is computed
        # from the damped palm to keep the drawing visually steady.
        if self._pinch_mode == "one" and self._pinch_anchor_palm is not None:
            active_palm = _palm_for_slot(self._pinch_one_slot)
            if active_palm is not None:
                live_dx = active_palm[0] - self._pinch_anchor_palm[0]
                live_dy = active_palm[1] - self._pinch_anchor_palm[1]
                self._emit_pinch_transform(
                    self._pinch_accum_dx + live_dx,
                    self._pinch_accum_dy + live_dy,
                    self._pinch_accum_scale,
                )
        elif (
            self._pinch_mode == "two"
            and primary_palm_s is not None
            and secondary_palm_s is not None
        ):
            cur_mid = (
                (primary_palm_s[0] + secondary_palm_s[0]) * 0.5,
                (primary_palm_s[1] + secondary_palm_s[1]) * 0.5,
            )
            cur_dist = max(
                1e-4,
                ((primary_palm_s[0] - secondary_palm_s[0]) ** 2
                 + (primary_palm_s[1] - secondary_palm_s[1]) ** 2) ** 0.5,
            )
            live_dx = cur_mid[0] - self._pinch_two_anchor_mid[0]
            live_dy = cur_mid[1] - self._pinch_two_anchor_mid[1]
            if self._pinch_two_anchor_dist > 1e-4:
                raw_ratio = cur_dist / self._pinch_two_anchor_dist
                # Same sensitivity gain as the lock-in path so the
                # live preview matches what gets baked when the
                # user releases.
                scale_factor = 1.0 + _PINCH_SCALE_SENSITIVITY * (raw_ratio - 1.0)
            else:
                scale_factor = 1.0
            self._emit_pinch_transform(
                self._pinch_accum_dx + live_dx,
                self._pinch_accum_dy + live_dy,
                self._pinch_accum_scale * scale_factor,
            )

        # Deferred smoother reset. Now that mode-transition lock-in
        # AND the live transform emit have BOTH consumed the
        # smoothed palm values, we can safely null the smoother
        # state if we ended the frame in 'none' mode. Doing this
        # earlier (the previous version did) skipped the lock-in
        # on the release frame and lost the user's drag â€” the
        # next pinch then snapped back to the pre-drag offset.
        if self._pinch_mode == "none":
            self._pinch_smoothed_primary = None
            self._pinch_smoothed_secondary = None

    def _emit_pinch_transform(self, dx: float, dy: float, scale: float) -> None:
        # Palm coords are in user-perspective normalised image space
        # (cv2.flip is applied before MediaPipe), so a swipe to the
        # right increases palm-x. The displayed overlay should move
        # in the same direction â†’ no axis inversion.
        try:
            self.drawing_overlay_grab_transform.emit(float(dx), float(dy), float(scale))
        except Exception:
            pass

    def _handle_app_controls(self, prediction, hand_reading, hand_handedness: str | None, now: float) -> None:
        if self._tutorial_mode_enabled:
            self._handle_tutorial_controls(prediction, hand_reading, hand_handedness, now)
            return

        # Apply the user's Gesture Binds remap before any handler reads
        # prediction.stable_label. If the bound action is custom, this
        # also fires it via _dispatch_action. See the helper docstring.
        prediction = self._apply_gesture_binding_remap(prediction, hand_handedness, now)

        # Mid-count 3→2→1 owns one/two/three. Do not arm Chrome,
        # Spotify, or YouTube holds from the builtin label — the
        # banner is not a fire, but those routers start timers as
        # soon as they see the name.
        if self._pose_sequence_owns_motion():
            sl = str(getattr(prediction, "stable_label", "") or "")
            if sl in {"one", "two", "three", "four"}:
                prediction = self._neutralize_prediction(prediction)

        # Default-bound open_chrome (right_three) / open_touchless
        # (right_four) actions have no router. Fire them here on a
        # 1.0 s hold. Runs BEFORE the rest of the handler so a
        # three / four opens its target before any other router
        # (spotify, chrome) gets a turn.
        self._maybe_fire_open_chrome_touchless(prediction, hand_handedness, now)

        # When a save-location prompt is awaiting input, give the left-hand voice handler
        # a chance to run before any mode-specific branch returns early. This keeps the
        # left-fist cancel gesture available even while drawing / mouse / volume mode is
        # still active underneath the prompt.
        if self._save_prompt_active:
            left_pred = self._left_hand_prediction
            if hand_handedness == "Left" and prediction is not None:
                left_pred = prediction
            if left_pred is not None:
                self._handle_left_hand_voice(left_pred, now)
            if self._save_prompt_active:
                return

        if self._utility_capture_selection_active:
            self._update_utility_capture_selection(hand_reading, hand_handedness)
            self._update_youtube_wheel(prediction=None, hand_reading=None, now=now, active=False)
            self._update_chrome_wheel(prediction=None, hand_reading=None, now=now, active=False)
            self._update_spotify_wheel(prediction=None, hand_reading=None, now=now, active=False)
            self._reset_voice_candidate(now)
            self.chrome_router.reset()
            self.spotify_router.reset()
            if self._drawing_mode_enabled:
                self._drawing_tool = "hidden"
                self._drawing_cursor_norm = None
                self._camera_draw_last_point = None
            return

        if self._utility_recording_active and hand_handedness == 'Right' and hand_reading is not None and self._utility_wheel_pose_active(hand_reading):
            if self._utility_recording_stop_candidate_since <= 0.0:
                self._utility_recording_stop_candidate_since = now
            elif now - self._utility_recording_stop_candidate_since >= 0.6:
                self._queue_utility_request('stop_recording')
                self.command_detected.emit('Stopping screen recording')
                self._utility_recording_stop_candidate_since = 0.0
                self._utility_wheel_cooldown_until = now + 1.2
            self._update_youtube_wheel(prediction=None, hand_reading=None, now=now, active=False)
            self._update_chrome_wheel(prediction=None, hand_reading=None, now=now, active=False)
            self._update_spotify_wheel(prediction=None, hand_reading=None, now=now, active=False)
            self._reset_voice_candidate(now)
            self.chrome_router.reset()
            self.spotify_router.reset()
            return
        self._utility_recording_stop_candidate_since = 0.0

        utility_wheel_consuming = self._update_utility_wheel(hand_reading, hand_handedness, now)
        if utility_wheel_consuming:
            self._update_youtube_wheel(prediction=None, hand_reading=None, now=now, active=False)
            self._update_chrome_wheel(prediction=None, hand_reading=None, now=now, active=False)
            self._update_spotify_wheel(prediction=None, hand_reading=None, now=now, active=False)
            self._reset_voice_candidate(now)
            self.chrome_router.reset()
            self.spotify_router.reset()
            if self._drawing_mode_enabled:
                self._drawing_tool = "hidden"
                self._drawing_cursor_norm = None
                self._camera_draw_last_point = None
            return

        if self._dictation_active:
            self._update_youtube_wheel(prediction=None, hand_reading=None, now=now, active=False)
            self._update_chrome_wheel(prediction=None, hand_reading=None, now=now, active=False)
            self._update_spotify_wheel(prediction=None, hand_reading=None, now=now, active=False)
            if self._left_hand_prediction is not None:
                self._handle_left_hand_voice(self._left_hand_prediction, now)
            else:
                self._reset_voice_candidate(now)
            return

        if self._volume_overlay_visible or self._volume_mode_active:
            self._update_youtube_wheel(prediction=None, hand_reading=None, now=now, active=False)
            self._update_chrome_wheel(prediction=None, hand_reading=None, now=now, active=False)
            self._update_spotify_wheel(prediction=None, hand_reading=None, now=now, active=False)
            if self._left_hand_prediction is not None:
                self._handle_left_hand_voice(self._left_hand_prediction, now)
            else:
                self._reset_voice_candidate(now)
            return

        drawing_toggle_consuming = self._handle_drawing_toggle(prediction, hand_handedness, now)
        if drawing_toggle_consuming:
            self._update_youtube_wheel(prediction=None, hand_reading=None, now=now, active=False)
            self._update_chrome_wheel(prediction=None, hand_reading=None, now=now, active=False)
            self._update_spotify_wheel(prediction=None, hand_reading=None, now=now, active=False)
            self._reset_voice_candidate(now)
            self.chrome_router.reset()
            self.spotify_router.reset()
            if not self._drawing_mode_enabled:
                self._reset_drawing_wheel(clear_cooldown=False)
            self._chrome_control_text = self._drawing_control_text
            self._spotify_control_text = self._drawing_control_text
            return

        if self._drawing_mode_enabled:
            self._update_youtube_wheel(prediction=None, hand_reading=None, now=now, active=False)
            self._update_chrome_wheel(prediction=None, hand_reading=None, now=now, active=False)
            self._update_spotify_wheel(prediction=None, hand_reading=None, now=now, active=False)
            self._reset_voice_candidate(now)
            self.chrome_router.reset()
            self.spotify_router.reset()
            self._reset_window_gesture_state(clear_cooldown=False)
            drawing_wheel_consuming = self._update_drawing_wheel(prediction, hand_reading, now, active=hand_handedness == "Right")
            if drawing_wheel_consuming:
                self._drawing_tool = "hidden"
                self._drawing_cursor_norm = None
                self._camera_draw_last_point = None
                self._chrome_control_text = self._drawing_control_text
                self._spotify_control_text = self._drawing_control_text
                return
            self._update_drawing_controls(prediction, hand_reading, hand_handedness, now)
            # Hover + swipe: left undoes the last mark, right clears.
            # Open-hand swipes fire while the pen is lifted; index-only
            # swipes still work as a fallback (and while drawing).
            if (
                hand_handedness == "Right"
                and hand_reading is not None
                and now >= self._drawing_swipe_cooldown_until
            ):
                one_pose_swipe = "neutral"
                try:
                    recog = getattr(self.engine, "dynamic_recognizer", None)
                    one_pose_swipe = str(
                        getattr(recog, "last_one_pose_horizontal_label", "neutral")
                        or "neutral"
                    )
                except Exception:
                    one_pose_swipe = "neutral"
                dyn = self._effective_dynamic_label(prediction)
                hovering = (
                    str(self._drawing_tool or "") == "hover"
                    or self._drawing_lift_pose_active(hand_reading)
                )
                swipe = "neutral"
                if hovering and dyn in {"swipe_left", "swipe_right"}:
                    swipe = dyn
                elif one_pose_swipe in {"swipe_left", "swipe_right"} and (
                    hovering
                    or self._drawing_draw_pose_active(prediction, hand_reading)
                ):
                    swipe = one_pose_swipe
                if swipe in {"swipe_left", "swipe_right"}:
                    if (
                        self._drawing_render_target == "camera"
                        and self._camera_draw_active_stroke_points
                    ):
                        self._commit_camera_draw_stroke()
                    self._camera_draw_last_point = None
                    self._drawing_draw_grace_until = 0.0
                    self._drawing_erase_grace_until = 0.0
                    self._drawing_draw_active_streak = 0
                    if self._perform_drawing_swipe_action(swipe):
                        self._drawing_swipe_cooldown_until = now + 1.2
                    self._chrome_control_text = self._drawing_control_text
                    self._spotify_control_text = self._drawing_control_text
                    return
            self._chrome_control_text = self._drawing_control_text
            self._spotify_control_text = self._drawing_control_text
            return

        if self._handle_window_control_gestures(hand_reading, hand_handedness, now):
            return

        # ---- Discord router (mode toggle + mute/deafen) ------------
        # Discord mode needs BOTH hands' stable labels every frame:
        # left hand for the mode-toggle activator, right hand for
        # mute / deafen actions. The engine processes one hand per
        # call to _handle_main_controls, so we synthesise both from
        # the current prediction + the cached _left_hand_prediction.
        left_pred = prediction if hand_handedness == "Left" else self._left_hand_prediction
        right_pred_label = (
            prediction.stable_label if hand_handedness == "Right" else "neutral"
        )
        # Discord activator is a CUSTOM left-hand pose: middle + ring
        # + pinky extended, index folded. The base recogniser labels
        # any three-finger pose as "three" so we'd collide with mouse
        # mode's left-3 toggle. The MRP geometry check disambiguates:
        # only the specific finger set fires "three_mrp", which is
        # the only label the Discord router accepts as its activator.
        left_reading_for_mrp = (
            hand_reading if hand_handedness == "Left" else self._left_hand_reading
        )
        # Discord integration is shelved behind a future feature update
        # (the Settings → General → Discord card renders only as a
        # "Coming soon" placeholder). To keep behaviour consistent we
        # force `left_mrp_active` off here so the MRP activator never
        # engages mouse-mode lockout, and we feed downstream code a
        # neutral snapshot so the router stays idle. The router itself
        # remains in the codebase for the future feature update — we're
        # just routing around it for now.
        left_mrp_active = False
        left_pred_label = (
            left_pred.stable_label if left_pred is not None else "neutral"
        )
        discord_snapshot = DiscordGestureSnapshot(
            control_text="",
            info_text="off",
            last_action="-",
            mode_active=False,
            consume_other_routes=False,
            suppress_mouse=False,
            action_counter=self._last_discord_action_counter,
        )
        self._discord_control_text = discord_snapshot.control_text
        # Emit a HUD toast on mode flip (4 s, same style as YouTube
        # mode notification) and record to the action history.
        if discord_snapshot.mode_active != self._discord_mode_prev_active:
            self._discord_mode_prev_active = discord_snapshot.mode_active
            label_text = (
                "Discord mode: ON" if discord_snapshot.mode_active else "Discord mode: OFF"
            )
            try:
                self.voice_status_overlay.show_info_hint(label_text, duration=4.0)
            except Exception:
                pass
            try:
                self.command_detected.emit(label_text)
            except Exception:
                pass
            try:
                self._record_action(
                    "discord_mode_on" if discord_snapshot.mode_active else "discord_mode_off",
                    label_text,
                )
            except Exception:
                pass
        # Emit a HUD toast each time a Discord mute/deafen action
        # fires — the user can't always see the Discord client, so
        # the toast confirms their gesture took effect.
        if discord_snapshot.action_counter != self._last_discord_action_counter:
            self._last_discord_action_counter = discord_snapshot.action_counter
            if discord_snapshot.last_action not in ("-", "discord_mode_on", "discord_mode_off"):
                try:
                    self.command_detected.emit(discord_snapshot.control_text)
                except Exception:
                    pass
                try:
                    self._record_action(
                        discord_snapshot.last_action, discord_snapshot.control_text
                    )
                except Exception:
                    pass
        # ------------------------------------------------------------

        # Mouse control is suppressed in two cases:
        #   1. Discord mode is currently active (user is on a call,
        #      wants free hand movement for mute/deafen without the
        #      cursor following along).
        #   2. The user is currently holding the Discord MRP activator
        #      pose. The base recogniser sees this as plain "three"
        #      which would otherwise also satisfy mouse-mode's left-3
        #      toggle — so we lock mouse-mode out for any frame where
        #      MRP is detected, regardless of whether Discord mode is
        #      already on. Without this gate the same gesture would
        #      simultaneously toggle Discord mode AND toggle mouse mode.
        if discord_snapshot.suppress_mouse or left_mrp_active:
            mouse_consuming = False
        else:
            mouse_consuming = self._handle_mouse_control(
                prediction=prediction,
                hand_reading=hand_reading if hand_handedness == "Right" else None,
                hand_handedness=hand_handedness,
                now=now,
            )
        if mouse_consuming:
            self._update_youtube_wheel(prediction=None, hand_reading=None, now=now, active=False)
            self._update_chrome_wheel(prediction=None, hand_reading=None, now=now, active=False)
            self._update_spotify_wheel(prediction=None, hand_reading=None, now=now, active=False)
            self._reset_voice_candidate(now)
            self.chrome_router.reset()
            self.spotify_router.reset()
            self._chrome_control_text = "Mouse mode active"
            self._spotify_control_text = "Mouse mode active"
            return

        if self._left_hand_prediction is not None:
            self._handle_left_hand_voice(self._left_hand_prediction, now)
            # CLIP GESTURE — runs alongside voice handler. Voice
            # handler returns early on fist/one/two; clip fires on
            # left-hand TWO so the two never collide.
            self._handle_left_hand_clip_gesture(
                self._left_hand_prediction,
                self._left_hand_reading,
                now,
            )
            if hand_handedness != "Right":
                self._update_youtube_wheel(prediction=None, hand_reading=None, now=now, active=False)
                self._update_chrome_wheel(prediction=None, hand_reading=None, now=now, active=False)
                self._update_spotify_wheel(prediction=None, hand_reading=None, now=now, active=False)
                return
        else:
            self._reset_voice_candidate(now)
        app_static_label = self._derive_app_static_label(prediction, hand_reading)
        right_hand_active = hand_handedness == "Right"

        youtube_wheel_consuming = self._update_youtube_wheel(
            prediction,
            hand_reading,
            now,
            active=right_hand_active,
        )
        if youtube_wheel_consuming:
            return

        chrome_wheel_consuming = self._update_chrome_wheel(
            prediction,
            hand_reading,
            now,
            active=right_hand_active,
        )
        if chrome_wheel_consuming:
            return

        spotify_wheel_consuming = self._update_spotify_wheel(
            prediction,
            hand_reading,
            now,
            active=right_hand_active,
        )
        if spotify_wheel_consuming:
            return

        youtube_snapshot = self.youtube_router.update(
            stable_label=app_static_label,
            dynamic_label=self._effective_dynamic_label(prediction),
            controller=self.youtube_controller,
            now=now,
        )
        self._youtube_control_text = youtube_snapshot.control_text
        self._youtube_mode_info = youtube_snapshot.info_text
        youtube_now_active = youtube_snapshot.info_text in {"forced", "auto"}
        if youtube_now_active != self._youtube_mode_prev_active:
            self._youtube_mode_prev_active = youtube_now_active
            label_text = "YouTube mode: ON" if youtube_now_active else "YouTube mode: OFF"
            try:
                self.voice_status_overlay.show_info_hint(label_text, duration=4.0)
            except Exception:
                pass
            try:
                self.command_detected.emit(label_text)
            except Exception:
                pass
            try:
                self._record_action(
                    "youtube_mode_on" if youtube_now_active else "youtube_mode_off",
                    label_text,
                )
            except Exception:
                pass
        if youtube_snapshot.action_counter != self._last_youtube_action_counter:
            self._last_youtube_action_counter = youtube_snapshot.action_counter
            if youtube_snapshot.last_action != "-":
                self.command_detected.emit(youtube_snapshot.control_text)
                self._record_action(youtube_snapshot.last_action, youtube_snapshot.control_text)

        if youtube_snapshot.consume_other_routes:
            return

        chrome_snapshot = self.chrome_router.update(
            stable_label=app_static_label,
            dynamic_label=self._effective_dynamic_label(prediction),
            controller=self.chrome_controller,
            now=now,
        )
        self._chrome_mode_enabled = chrome_snapshot.mode_enabled
        self._chrome_control_text = chrome_snapshot.control_text
        chrome_now_active = bool(chrome_snapshot.mode_enabled)
        if chrome_now_active != self._chrome_mode_prev_active:
            self._chrome_mode_prev_active = chrome_now_active
            try:
                self.voice_status_overlay.show_info_hint(
                    "Chrome mode: ON" if chrome_now_active else "Chrome mode: OFF",
                    duration=3.0,
                )
            except Exception:
                pass
        if chrome_snapshot.action_counter != self._last_chrome_action_counter:
            self._last_chrome_action_counter = chrome_snapshot.action_counter
            if chrome_snapshot.last_action != "-":
                self.command_detected.emit(chrome_snapshot.control_text)
                self._record_action(chrome_snapshot.last_action, chrome_snapshot.control_text)

        if chrome_snapshot.consume_other_routes or app_static_label in {"three", "three_together"}:
            return

        if youtube_snapshot.mode_active:
            return

        snapshot = self.spotify_router.update(
            stable_label=prediction.stable_label,
            dynamic_label=self._effective_dynamic_label(prediction),
            controller=self.spotify_controller,
            now=now,
        )
        self._spotify_control_text = snapshot.control_text
        self._spotify_info_text = snapshot.info_text
        if snapshot.action_counter != self._last_spotify_action_counter:
            self._last_spotify_action_counter = snapshot.action_counter
            if snapshot.last_action != "-":
                self._last_spotify_action = snapshot.last_action
                self.command_detected.emit(snapshot.control_text)
                self._record_action(snapshot.last_action, snapshot.control_text)

    def _request_spotify_connect_prompt(self) -> None:
        """Throttled bridge: the SpotifyController opened-listener fires this
        when Spotify is opened by gesture/voice while the Web API is NOT
        connected. Emit the UI signal at most once per cooldown so a burst of
        opens (or the repeating 'two' focus) doesn't stack pills. May be called
        from the voice worker thread — the Qt signal marshals to the GUI thread."""
        try:
            now = time.monotonic()
            if now < self._spotify_connect_prompt_cooldown_until:
                return
            self._spotify_connect_prompt_cooldown_until = now + 30.0
            self.spotify_connect_prompt_requested.emit()
        except Exception:
            pass

    def _handle_tutorial_controls(self, prediction, hand_reading, hand_handedness: str | None, now: float) -> None:
        step_key = self._tutorial_step_key or ""

        utility_wheel_consuming = self._update_utility_wheel(hand_reading, hand_handedness, now)
        if utility_wheel_consuming:
            self._update_youtube_wheel(prediction=None, hand_reading=None, now=now, active=False)
            self._update_chrome_wheel(prediction=None, hand_reading=None, now=now, active=False)
            self._update_spotify_wheel(prediction=None, hand_reading=None, now=now, active=False)
            self._reset_voice_candidate(now)
            self.chrome_router.reset()
            self.spotify_router.reset()
            if self._drawing_mode_enabled:
                self._drawing_tool = "hidden"
                self._drawing_cursor_norm = None
                self._camera_draw_last_point = None
            return

        if self._dictation_active:
            self._update_youtube_wheel(prediction=None, hand_reading=None, now=now, active=False)
            self._update_chrome_wheel(prediction=None, hand_reading=None, now=now, active=False)
            self._update_spotify_wheel(prediction=None, hand_reading=None, now=now, active=False)
            if step_key == "voice_command" and self._left_hand_prediction is not None:
                self._handle_left_hand_voice(self._left_hand_prediction, now)
            else:
                self._reset_voice_candidate(now)
            return

        if step_key == "mouse_mode":
            mouse_consuming = self._handle_mouse_control(
                prediction=prediction,
                hand_reading=hand_reading if hand_handedness == "Right" else None,
                hand_handedness=hand_handedness,
                now=now,
            )
            self._update_youtube_wheel(prediction=None, hand_reading=None, now=now, active=False)
            self._update_chrome_wheel(prediction=None, hand_reading=None, now=now, active=False)
            self._update_spotify_wheel(prediction=None, hand_reading=None, now=now, active=False)
            self._reset_voice_candidate(now)
            self.chrome_router.reset()
            self.spotify_router.reset()
            if mouse_consuming:
                self._chrome_control_text = "Mouse mode active"
                self._spotify_control_text = "Mouse mode active"
            return

        self._reset_voice_candidate(now)
        self._update_youtube_wheel(prediction=None, hand_reading=None, now=now, active=False)
        self._update_chrome_wheel(prediction=None, hand_reading=None, now=now, active=False)

        if self._mouse_mode_enabled:
            self._update_spotify_wheel(prediction=None, hand_reading=None, now=now, active=False)
            self.chrome_router.reset()
            self.spotify_router.reset()
            self._chrome_control_text = "Mouse mode active"
            self._spotify_control_text = "Mouse mode active"
            return

        if step_key == "voice_command":
            self._update_spotify_wheel(prediction=None, hand_reading=None, now=now, active=False)
            if self._left_hand_prediction is not None:
                self._handle_left_hand_voice(self._left_hand_prediction, now)
            return

        if hand_handedness != "Right":
            self._update_spotify_wheel(prediction=None, hand_reading=None, now=now, active=False)
            self.chrome_router.reset()
            self.spotify_router.reset()
            return

        if step_key == "gesture_wheel":
            self.chrome_router.reset()
            self.spotify_router.reset()
            self._update_spotify_wheel(
                prediction,
                hand_reading,
                now,
                active=hand_reading is not None,
            )
            return

        self._update_spotify_wheel(prediction=None, hand_reading=None, now=now, active=False)
        self.chrome_router.reset()

        if step_key == "swipes":
            self.spotify_router.reset()
            self._chrome_control_text = "tutorial swipe practice"
            self._spotify_control_text = "tutorial swipe practice"
            return

        stable_label = str(getattr(prediction, "stable_label", "neutral") or "neutral")
        if step_key == "spotify_open":
            routed_label = "two" if stable_label == "two" else "neutral"
        elif step_key == "play_pause":
            routed_label = "fist" if stable_label == "fist" else "neutral"
        else:
            self.spotify_router.reset()
            return

        snapshot = self.spotify_router.update(
            stable_label=routed_label,
            dynamic_label="neutral",
            controller=self.spotify_controller,
            now=now,
        )
        self._spotify_control_text = snapshot.control_text
        self._spotify_info_text = snapshot.info_text
        if snapshot.action_counter != self._last_spotify_action_counter:
            self._last_spotify_action_counter = snapshot.action_counter
            if snapshot.last_action != "-":
                self._last_spotify_action = snapshot.last_action
                self.command_detected.emit(snapshot.control_text)
                self._record_action(snapshot.last_action, snapshot.control_text)

    def _any_gesture_wheel_visible(self) -> bool:
        """True when any radial gesture-wheel UI is up. Used to gate
        the OS cursor so that the user's directional hand motion
        (selecting a wedge) doesn't also drag the desktop cursor."""
        return bool(
            getattr(self, "_chrome_wheel_visible", False)
            or getattr(self, "_spotify_wheel_visible", False)
            or getattr(self, "_youtube_wheel_visible", False)
            or getattr(self, "_drawing_wheel_visible", False)
            or getattr(self, "_utility_wheel_visible", False)
        )

    def _handle_mouse_control(self, prediction, hand_reading, hand_handedness: str | None, now: float) -> bool:
        # Tutorial isolation: in tutorial mode, only the mouse_mode
        # step is allowed to engage the mouse pipeline at all. On
        # every other step we reset the tracker + bail so the
        # user's left-three hold (or any other gesture) can't
        # silently toggle mouse mode while they're learning a
        # different gesture. The mouse_mode step itself runs the
        # full pipeline but suppresses OS-level cursor / clicks
        # via tutorial_demo_only further down.
        if (
            self._tutorial_mode_enabled
            and self._tutorial_step_key != "mouse_mode"
        ):
            self.mouse_tracker.reset()
            self._last_mouse_update = self._blank_mouse_update()
            self._mouse_mode_enabled = False
            return False

        if not self.mouse_controller.available:
            self.mouse_tracker.reset()
            self._last_mouse_update = self._blank_mouse_update()
            self._mouse_mode_enabled = False
            self._mouse_status_text = "unavailable"
            self._mouse_control_text = self.mouse_controller.message
            return False

        # When the user has constrained mouse mode to a single
        # monitor, the camera-frame red box should reflect THAT
        # monitor's aspect, not the full virtual desktop's. Without
        # this, picking Monitor 1 on a side-by-side dual setup
        # leaves the box at the desktop's 32:9 aspect (super-wide,
        # mostly empty padding around a single hand) — looks weird
        # because the visualization shows a single monitor but the
        # box is sized for two. Honor the saved index here so the
        # tracker computes a tighter, monitor-sized box.
        active_monitor_idx = getattr(self.config, "mouse_active_monitor_index", None)
        chosen_bounds = None
        if isinstance(active_monitor_idx, int):
            try:
                from PySide6.QtGui import QGuiApplication
                screens = list(QGuiApplication.screens() or [])
                if 0 <= active_monitor_idx < len(screens):
                    geo = screens[active_monitor_idx].geometry()
                    chosen_bounds = (geo.x(), geo.y(), geo.width(), geo.height())
            except Exception:
                chosen_bounds = None
        self.mouse_tracker.set_desktop_bounds(
            chosen_bounds if chosen_bounds is not None else self.mouse_controller.virtual_bounds()
        )
        update = self.mouse_tracker.update(
            hand_reading=hand_reading,
            prediction=prediction,
            hand_handedness=hand_handedness,
            cursor_seed=self.mouse_controller.current_position_normalized(),
            now=now,
        )
        self._last_mouse_update = update

        # During the tutorial's mouse_mode step, suppress every
        # OS-level mouse output (cursor move + click + scroll). The
        # tutorial overlay still shows the cursor dot driven by
        # update.cursor_position, so the user gets the visual
        # feedback for free without the OS cursor flying around
        # while they're trying to interact with the tutorial UI.
        # Outside the tutorial, mouse control fires normally on
        # whatever app the user is focused on (including Touchless
        # itself — Win32 SetCursorPos is global and not gated by
        # focus).
        tutorial_demo_only = (
            self._tutorial_mode_enabled
            and self._tutorial_step_key == "mouse_mode"
        )
        # Gesture wheels (chrome / spotify / youtube / drawing /
        # utility) need the OS cursor to FREEZE while the user is
        # picking a wedge: the user moves their hand directionally
        # to select a slice, and without this gate that hand motion
        # also drags the desktop cursor across the screen, leaving
        # them on a random app the moment they release the pose.
        # The wheel overlay already shows a local pointer inside the
        # ring, so the user has visual feedback without OS cursor
        # movement. Clicks/scrolls stay gated only by tutorial_demo_only
        # — the wheel itself owns selection, so a stray pinch/scroll
        # during wheel pose shouldn't reach the OS either, but the
        # mouse-tracker's update typically won't produce them anyway
        # when the active pose is the wheel-pose (not pinch).
        wheel_locked = self._any_gesture_wheel_visible()
        if not tutorial_demo_only:
            if update.cursor_position is not None and not wheel_locked:
                cx, cy = self._cursor_to_active_monitor(*update.cursor_position)
                self.mouse_controller.move_normalized(cx, cy)
            if update.left_press:
                self.mouse_controller.left_down()
            if update.left_release:
                self.mouse_controller.left_up()
            if update.left_click:
                self.mouse_controller.left_click()
            if update.right_click:
                self.mouse_controller.right_click()
            if update.scroll_steps:
                self.mouse_controller.scroll(update.scroll_steps)

        prior_mode_enabled = self._mouse_mode_enabled
        self._mouse_mode_enabled = update.mode_enabled
        self._mouse_status_text = update.status
        # Toggle pill: matches the Drawing-mode pattern at
        # _toggle_drawing_mode â€” same VoiceStatusOverlay info hint
        # so the user gets identical visual feedback when they
        # flip mouse mode on or off via the gesture.
        if update.mode_enabled != prior_mode_enabled:
            try:
                self.voice_status_overlay.show_info_hint(
                    "Mouse Mode: On" if update.mode_enabled else "Mouse Mode: Off",
                    duration=3.0,
                )
            except Exception:
                pass
            # Mirror the action history + telemetry that the static
            # binding path produces, so gesture-driven mode toggles
            # are visible alongside everything else. The label uses
            # `mouse_mode_on/off` to distinguish from the bound-hold
            # `mouse_mode_toggle` action_id, but you can collapse them
            # in the dashboard if useful.
            self._record_action(
                "mouse_mode_on" if update.mode_enabled else "mouse_mode_off",
                "Mouse mode on" if update.mode_enabled else "Mouse mode off",
            )
            # On the off->on transition only, emit the activation
            # signal so the main window can show the "which monitor
            # to control" picker. The signal is queued (Qt
            # auto-queues across threads) so the dialog construction
            # happens cleanly on the GUI thread.
            if update.mode_enabled:
                try:
                    self.mouse_mode_activated.emit()
                except Exception:
                    pass
            # Log to recent-actions so the user sees gesture-driven
            # mode toggles in the action history (was previously
            # only logging media + volume actions).
            self._record_action(
                "mouse_mode_on" if update.mode_enabled else "mouse_mode_off",
                "Mouse mode on" if update.mode_enabled else "Mouse mode off",
            )
        action_text = update.control_text
        if not tutorial_demo_only and (update.left_press or update.left_release or update.left_click or update.right_click or update.scroll_steps):
            action_text = self.mouse_controller.message
            self.command_detected.emit(action_text)
        self._mouse_control_text = action_text
        return update.consume_other_routes

    def _update_volume_overlay(self) -> None:
        msg = self._volume_message if self._volume_mode_active else self._volume_status_text
        if self._volume_dual_active:
            self.volume_overlay.set_dual_level(
                self._volume_app_level,
                self._volume_level,
                self._volume_app_label,
                muted=self._volume_muted,
                active=self._volume_mode_active,
                selected_bar=self._volume_bar_selected,
                message=msg,
            )
        else:
            self.volume_overlay.set_level(
                self._volume_level,
                muted=self._volume_muted,
                active=self._volume_mode_active,
                message=msg,
            )
        if self._volume_overlay_visible:
            if not self.volume_overlay.isVisible():
                self.volume_overlay.show_overlay()
        elif self.volume_overlay.isVisible():
            self.volume_overlay.hide_overlay()

    def _is_significant_state_change(self, result) -> bool:
        # Returns True when this tick has a viewer-relevant change
        # that we should emit even if we're in the throttled-skip
        # half of the cycle: a hand appearing or disappearing, or
        # a new gesture firing. The user perceives those events as
        # "the app reacted instantly"; we don't want a 30 fps emit
        # cap to delay them by ~16 ms. Pure-motion frames (hand
        # already visible, no action change) can ride the throttle.
        try:
            current_found = bool(result.found)
        except Exception:
            return True
        if current_found != self._last_result_had_hand:
            return True
        if self._action_history_dirty_for_emit:
            self._action_history_dirty_for_emit = False
            return True
        return False

    def _read_system_mute(self) -> bool:
        # Cache the COM IAudioEndpointVolume.GetMute call for ~120 ms
        # â€” was firing every frame (1-3 ms each) inside _handle_volume
        # _control's hot path. The mute state can't change between
        # gesture frames in any meaningful way (the user can't
        # press the mute key during a swipe), so refresh-rate is
        # plenty for UX purposes.
        now = time.monotonic()
        if now < self._mute_cache_until:
            return self._mute_cache_value
        muted = self.volume_controller.get_mute()
        self._mute_cache_value = bool(muted) if muted is not None else False
        self._mute_cache_until = now + 0.12
        return self._mute_cache_value

    def _queue_spotify_volume(self, volume_percent: int) -> None:
        volume_percent = max(0, min(100, int(volume_percent)))
        with self._spotify_vol_lock:
            self._spotify_vol_target = volume_percent
            worker = self._spotify_vol_worker
            if worker is not None and worker.is_alive():
                return
            self._spotify_vol_worker = threading.Thread(
                target=self._spotify_vol_worker_loop,
                daemon=True,
            )
            self._spotify_vol_worker.start()

    def _spotify_vol_worker_loop(self) -> None:
        min_interval = 0.12
        while True:
            with self._spotify_vol_lock:
                target = self._spotify_vol_target
                last = self._spotify_vol_last_sent
                if target is None or target == last:
                    self._spotify_vol_worker = None
                    return
                self._spotify_vol_last_sent = target
            try:
                self.spotify_controller.set_volume(target)
            except Exception:
                pass
            time.sleep(min_interval)

    def _update_runtime_status(self) -> None:
        if self._dictation_active:
            self._emit_status("dictation active")
        elif self._voice_listening:
            self._emit_status("voice listening...")
        elif self._drawing_mode_enabled:
            self._emit_status("Touchless active | drawing mode on")
        elif self._chrome_mode_enabled:
            self._emit_status("Touchless active | chrome mode on")
        else:
            self._emit_status("Touchless active")

    def _emit_status(self, text: str) -> None:
        normalized = str(text or "").strip()
        if not normalized or normalized == self._last_status_text:
            return
        self._last_status_text = normalized
        self.status_changed.emit(normalized)

    def _build_debug_payload(self, result, now: float) -> dict:
        prediction = result.prediction
        # Telemetry: emit `gesture_detected` on label TRANSITIONS so
        # holding a pose doesn't spam an event every frame. Use the
        # raw prediction labels (not the display-suppressed ones) so
        # we capture what the recognizer actually saw, even when
        # drawing mode is masking the chip. Tracks BOTH primary
        # (right) and secondary (left) hand predictions because
        # left-hand gestures (e.g. "four" → YouTube mode, pinch →
        # drawing) would otherwise be invisible to telemetry.
        #
        # Note: the recognizer's `stable_label` includes motion-
        # coupled poses (volume_pose, pinch, wheel_pose) — held
        # shapes that the user actually uses for continuous motion
        # control. We reclassify these as kind="dynamic" so the
        # dashboard's Held vs Motion split matches the user's mental
        # model from the control guide.
        _MOTION_COUPLED_STABLE = {"volume_pose", "pinch", "wheel_pose"}
        try:
            in_tutorial = bool(getattr(self, "_tutorial_mode_enabled", False))
            primary_handedness = ""
            if result.found and result.tracked_hand is not None:
                primary_handedness = str(result.tracked_hand.handedness or "").lower()
            secondary_handedness = "left" if primary_handedness == "right" else (
                "right" if primary_handedness == "left" else ""
            )
            secondary_pred = getattr(result, "secondary_prediction", None)

            in_drawing = bool(getattr(self, "_drawing_mode_enabled", False))
            in_mouse = bool(getattr(self, "_mouse_mode_enabled", False))
            DEBOUNCE = 0.75  # seconds; collapses recognizer flicker

            def _emit_transition(pred, kind_attr, last_attr, kind_default, hand_label):
                if pred is None:
                    return
                value = str(getattr(pred, kind_attr, "neutral") or "neutral")
                last = getattr(self, last_attr, "neutral")
                if value != last and value not in ("", "neutral"):
                    debounce_key = (hand_label, value)
                    last_emit = self._telemetry_gesture_last_emit.get(debounce_key, 0.0)
                    if now - last_emit >= DEBOUNCE:
                        self._telemetry_gesture_last_emit[debounce_key] = now
                        kind = kind_default
                        if kind_default == "static" and (
                            value in _MOTION_COUPLED_STABLE or value.startswith("swipe_")
                        ):
                            kind = "dynamic"
                        from ...telemetry import track as _track
                        _track(
                            "gesture_detected",
                            {
                                "gesture": value,
                                "kind": kind,
                                "handedness": hand_label,
                                "in_tutorial": in_tutorial,
                                "in_drawing_mode": in_drawing,
                                "in_mouse_mode": in_mouse,
                            },
                        )
                setattr(self, last_attr, value)

            _emit_transition(prediction,    "stable_label",  "_telemetry_last_static_label",            "static",  primary_handedness)
            _emit_transition(prediction,    "dynamic_label", "_telemetry_last_dynamic_label",           "dynamic", primary_handedness)
            _emit_transition(secondary_pred, "stable_label",  "_telemetry_last_static_label_secondary",  "static",  secondary_handedness)
            _emit_transition(secondary_pred, "dynamic_label", "_telemetry_last_dynamic_label_secondary", "dynamic", secondary_handedness)
        except Exception:
            pass

        # Suppress repeat_circle while drawing mode is on. The user
        # is actively drawing strokes with their index finger, and
        # the dynamic recogniser can label that motion as
        # repeat_circle as the index sweeps. No action fires on
        # repeat_circle in drawing mode (the drawing branch returns
        # before reaching the routers), but the label was still
        # surfacing in the debug display and on _dynamic_hold_label,
        # which the user found distracting. Swipes are kept because
        # drawing mode actively uses left/right swipes for nav.
        incoming_dynamic = self._effective_dynamic_label(prediction)
        if self._drawing_mode_enabled and incoming_dynamic == "repeat_circle":
            incoming_dynamic = "neutral"
        dynamic_display = incoming_dynamic
        if incoming_dynamic != "neutral":
            self._dynamic_hold_label = incoming_dynamic
            self._dynamic_hold_until = now + 0.85
        elif now < self._dynamic_hold_until:
            dynamic_display = self._dynamic_hold_label
        else:
            self._dynamic_hold_label = "neutral"
            self._dynamic_hold_until = 0.0

        payload_raw_label = prediction.raw_label
        payload_stable_label = prediction.stable_label
        payload_dynamic_label = dynamic_display
        banner_text = prediction.stable_label if prediction.stable_label != "neutral" else prediction.raw_label
        if banner_text in self._UNNAMED_RECOGNIZER_LABELS:
            banner_text = "neutral"
        seq_banner = None
        try:
            seq = getattr(self, "_pose_sequence_runtime", None)
            if seq is not None:
                seq_banner = seq.current_banner()
        except Exception:
            seq_banner = None
        if self._drawing_mode_enabled:
            payload_raw_label = "neutral"
            payload_stable_label = "neutral"
            payload_dynamic_label = "neutral"
            if self._drawing_tool == "draw":
                gesture_chip = "Drawing: draw"
            elif self._drawing_tool == "erase":
                gesture_chip = "Drawing: erase"
            elif self._drawing_tool == "hover":
                gesture_chip = "Drawing: hover"
            else:
                gesture_chip = "Drawing mode on"
        elif self._volume_mode_active:
            # While volume control is held, force the chip to stay
            # on a stable "Volume" label. The static recognizer
            # itself flickers between 'volume_pose', 'two', and
            # 'neutral' frame-to-frame even when the tracker is
            # solidly active (the tracker has its own structural
            # gate independent of the recognizer's label), so the
            # chip used to flip on/off once a second instead of
            # showing the user that volume control is engaged.
            gesture_chip = "Volume"
        elif seq_banner:
            gesture_chip = f"Gesture: {seq_banner[0]}"
        elif dynamic_display != "neutral":
            gesture_chip = f"Dynamic: {dynamic_display.replace('_', ' ')}"
        else:
            gesture_chip = f"Gesture: {banner_text}"

        handedness = ""
        if result.found and result.tracked_hand is not None:
            handedness = str(result.tracked_hand.handedness or "").lower()

        wheel_items = ()
        wheel_selected_key = None
        wheel_selection_progress = 0.0
        wheel_cursor_offset = None
        wheel_kind = "none"
        wheel_visible = False
        if self._chrome_wheel_visible:
            wheel_kind = "chrome"
            wheel_visible = True
            wheel_items = self._chrome_wheel_items()
            wheel_selected_key = self._chrome_wheel_selected_key
            wheel_cursor_offset = self._chrome_wheel_cursor_offset
            if self._chrome_wheel_selected_key is not None:
                wheel_selection_progress = min(1.0, max(0.0, (now - self._chrome_wheel_selected_since) / 1.0))
        elif self._spotify_wheel_visible:
            wheel_kind = "spotify"
            wheel_visible = True
            wheel_items = self._spotify_wheel_items()
            wheel_selected_key = self._spotify_wheel_selected_key
            wheel_cursor_offset = self._spotify_wheel_cursor_offset
            if self._spotify_wheel_selected_key is not None:
                wheel_selection_progress = min(1.0, max(0.0, (now - self._spotify_wheel_selected_since) / 1.0))

        info_lines = [
            f"Camera: {self._camera_info.display_name if self._camera_info is not None else 'waiting'}",
            "Handedness: -",
            f"Gesture raw/stable: {prediction.raw_label} / {prediction.stable_label}",
            f"Confidence: {prediction.confidence:.2f}",
            f"FPS: {self._fps:.1f}",
            "Box: -",
            "Palm: -",
            f"Dynamic: {dynamic_display.replace('_', ' ') if dynamic_display != 'neutral' else 'neutral'}",
            "Candidates: -",
            "Thumb: -",
            "Index: -",
            "Middle: -",
            "Ring: -",
            "Pinky: -",
            "Spreads: -",
            "Reasoning: no hand in frame",
            f"Volume control: {self._volume_message}",
            self._format_volume_level_text(),
            f"Spotify control: {self._spotify_control_text}",
            f"Spotify info: {self._spotify_info_text}",
            f"Chrome mode: {'on' if self._chrome_mode_enabled else 'off'}",
            f"Chrome control: {self._chrome_control_text}",
            f"YouTube mode: {self._youtube_mode_info}",
            f"YouTube control: {self._youtube_control_text}",
            f"Voice mode: {self._voice_mode_text()}",
            f"Voice control: {self._voice_control_text}",
            f"Voice heard: {self._voice_preview_text(self._voice_display_text)}",
            f"Mouse mode: {'on' if self._mouse_mode_enabled else 'off'} ({self._mouse_status_text})",
            f"Mouse control: {self._mouse_control_text}",
        ]
        if self._window_pair_overlay is not None:
            info_lines.append(f"Window pair distance: {float(self._window_pair_overlay.get('distance', 0.0) or 0.0):.2f}")
        info_lines.append(f"Gestures: {'on' if self._gestures_enabled else 'off'}")
        info_lines.append(f"Drawing: {'on' if self._drawing_mode_enabled else 'off'} | {self._drawing_control_text} | target={self._drawing_render_target}")

        if result.found and result.tracked_hand is not None and result.hand_reading is not None:
            hand = result.tracked_hand
            reading = result.hand_reading
            info_lines[1] = f"Handedness: {hand.handedness} ({hand.handedness_confidence:.2f})"
            info_lines[5] = (
                f"Box: x={hand.bbox.x:.2f} y={hand.bbox.y:.2f} "
                f"w={hand.bbox.width:.2f} h={hand.bbox.height:.2f}"
            )
            info_lines[6] = (
                f"Palm: roll={reading.palm.roll_deg:.1f} "
                f"pitch={reading.palm.pitch_deg:.1f} yaw={reading.palm.yaw_deg:.1f}"
            )
            info_lines[7] = f"Dynamic: {dynamic_display.replace('_', ' ') if dynamic_display != 'neutral' else 'neutral'}"
            candidate_text = ", ".join(
                f"{candidate.label}={candidate.score:.2f}" for candidate in prediction.candidates[:4]
            ) or "-"
            info_lines[8] = f"Candidates: {candidate_text}"
            for row, name in zip(range(9, 14), ("thumb", "index", "middle", "ring", "pinky")):
                finger = reading.fingers[name]
                info_lines[row] = (
                    f"{name.title()}: {finger.state} | "
                    f"open={finger.openness:.2f} curl={finger.curl:.2f} "
                    f"conf={finger.confidence:.2f} occ={finger.occluded}"
                )
            spread_text = ", ".join(
                f"{name}={spread.state}:{spread.distance:.2f}" for name, spread in reading.spreads.items()
            ) or "-"
            info_lines[14] = f"Spreads: {spread_text}"
            info_lines[15] = (
                f"Reasoning: extended={reading.finger_count_extended} "
                f"occlusion={reading.occlusion_score:.2f} shape={reading.shape_confidence:.2f}"
            )
        # Raw per-frame static-pose scores from the recognizer — top-N
        # by score, capped to 3 for display. The recognizer maintains
        # `last_static_scores` as a dict of label→confidence updated
        # each frame; the Settings → General → Diagnostics 'top
        # scores' toggle renders this in the home tracking pill so the
        # user can see runner-up poses (the ones that ALMOST won).
        try:
            scores_dict = self.engine.last_static_scores if self.engine is not None else {}
            top_scores = sorted(
                scores_dict.items(), key=lambda kv: float(kv[1]), reverse=True
            )[:3]
        except Exception:
            top_scores = []

        return {
            "result": result,
            "gesture_chip": gesture_chip,
            "info_lines": info_lines,
            "found": bool(result.found),
            "handedness": handedness,
            "raw_label": payload_raw_label,
            "stable_label": payload_stable_label,
            "dynamic_label": payload_dynamic_label,
            "confidence": float(prediction.confidence),
            "top_static_scores": top_scores,
            "fps": float(self._fps),
            "low_fps_active": bool(self._low_fps_active),
            "low_fps_forced": bool(getattr(self.config, "low_fps_mode", False)),
            "low_fps_auto_engaged": bool(self._low_fps_auto_engaged),
            "force_ten_fps_test_mode": bool(getattr(self.config, "force_ten_fps_test_mode", False)),
            # Was: is_window_active() or is_running(). is_running()
            # walks psutil.process_iter() with NO cache â€” per camera
            # frame that's a full Windows process-list scan, and the
            # walk gets heavier when Spotify is open (Spotify spawns
            # 5-10 helper processes), which matched the user-reported
            # 'camera lags while Spotify is open' regression. Switch
            # to is_window_open(), which already uses the 1-second
            # _spotify_window_handles cache and answers the same
            # question (is there a visible Spotify top-level window).
            "spotify_window_open": bool(self.spotify_controller.is_window_open()),
            # Authorization flag exposed so MainWindow can fire the
            # first-active prompt the moment a Spotify gesture / voice
            # command is attempted on an unauthorized install — not
            # just when Spotify is detected running. False both when
            # the user has never connected AND when the saved token
            # was revoked.
            "spotify_has_authorization": bool(getattr(self.spotify_controller, "has_authorization", False)),
            "spotify_control_text": self._spotify_control_text,
            "spotify_last_action": self._last_spotify_action,
            "chrome_mode_enabled": bool(self._chrome_mode_enabled),
            "chrome_window_active": bool(self.chrome_controller.is_window_active()),
            "chrome_window_open": bool(self.chrome_controller.is_window_open()),
            "chrome_control_text": self._chrome_control_text,
            "chrome_last_action": self._last_chrome_action,
            "voice_mode_text": self._voice_mode_text(),
            "voice_listening": bool(self._voice_listening),
            "voice_control_text": self._voice_control_text,
            "voice_heard_text": self._voice_preview_text(self._voice_display_text),
            "mouse_mode_enabled": bool(self._mouse_mode_enabled),
            "mouse_cursor_position": self.mouse_tracker.debug_state.cursor_position,
            # Camera-space bounds for the red mouse-control box +
            # virtual-desktop bounds for the monitor-map overlay.
            # Tutorial uses these so the cv2-side overlay can match
            # what the live view shows when running outside the
            # tutorial. None when mouse mode is off so the
            # consumer can skip drawing.
            "mouse_camera_control_bounds": (
                tuple(float(v) for v in self.mouse_tracker.debug_state.camera_control_bounds)
                if self.mouse_tracker.debug_state.camera_control_bounds is not None
                else None
            ),
            "mouse_virtual_bounds": (
                tuple(int(v) for v in self.mouse_controller.virtual_bounds())
                if getattr(self.mouse_controller, "available", False)
                else None
            ),
            "mouse_left_click": bool(getattr(self._last_mouse_update, "left_click", False)),
            "mouse_left_press": bool(getattr(self._last_mouse_update, "left_press", False)),
            # Scroll-step count emitted by the mouse_gesture tracker.
            # Positive = scroll up, negative = scroll down. The
            # tutorial's mouse-scroll practice arena reads this to
            # drive a QScrollArea instead of firing OS wheel events
            # (so the tutorial's scroll detection works regardless
            # of where the real cursor sits on the desktop).
            "mouse_scroll_steps": int(getattr(self._last_mouse_update, "scroll_steps", 0)),
            "gestures_enabled": bool(self._gestures_enabled),
            "drawing_mode_enabled": bool(self._drawing_mode_enabled),
            "drawing_tool": self._drawing_tool,
            "drawing_cursor_norm": self._drawing_cursor_norm,
            "drawing_control_text": self._drawing_control_text,
            "drawing_render_target": self._drawing_render_target,
            "drawing_request_token": int(self._drawing_request_token),
            "drawing_request_action": self._drawing_request_action,
            "drawing_shape_mode": bool(self._drawing_shape_mode),
            "utility_request_token": int(self._utility_request_token),
            "utility_request_action": self._utility_request_action,
            "utility_capture_selection_active": bool(self._utility_capture_selection_active),
            "utility_capture_cursor_norm": self._utility_capture_cursor_norm,
            "utility_capture_left_down": bool(self._utility_capture_left_down),
            "utility_capture_right_down": bool(self._utility_capture_right_down),
            "utility_recording_active": bool(self._utility_recording_active),
            "wheel_kind": wheel_kind,
            "wheel_visible": wheel_visible,
            "wheel_items": wheel_items,
            "wheel_selected_key": wheel_selected_key,
            "wheel_selection_progress": wheel_selection_progress,
            "wheel_cursor_offset": wheel_cursor_offset,
            "volume_level_scalar": self._volume_level,
            "volume_muted": self._volume_muted,
            "volume_active": self._volume_mode_active,
        }

    def _format_volume_level_text(self) -> str:
        mute_suffix = " [muted]" if self._volume_muted else ""
        if self._volume_level is None:
            return f"Volume level: -   ({self._volume_status_text}{mute_suffix})"
        return f"Volume level: {int(round(self._volume_level * 100))}%   ({self._volume_status_text}{mute_suffix})"

    def _volume_features_from_hand_reading(self, hand_reading):
        fine_states = {name: finger.state for name, finger in hand_reading.fingers.items()}
        states = {
            name: ("open" if finger.state in {"fully_open", "partially_curled"} else "closed")
            for name, finger in hand_reading.fingers.items()
        }
        open_scores = {name: float(finger.openness) for name, finger in hand_reading.fingers.items()}
        spread_states = {name: spread.state for name, spread in hand_reading.spreads.items()}
        spread_ratios = {name: float(spread.distance) for name, spread in hand_reading.spreads.items()}
        spread_together_strengths = {name: float(spread.together_strength) for name, spread in hand_reading.spreads.items()}
        spread_apart_strengths = {name: float(spread.apart_strength) for name, spread in hand_reading.spreads.items()}
        return SimpleNamespace(
            palm_scale=float(hand_reading.palm.scale),
            open_scores=open_scores,
            states=states,
            fine_states=fine_states,
            finger_count_open=sum(1 for finger in hand_reading.fingers.values() if finger.extended),
            spread_states=spread_states,
            spread_ratios=spread_ratios,
            spread_together_strengths=spread_together_strengths,
            spread_apart_strengths=spread_apart_strengths,
        )

    def _derive_app_static_label(self, prediction, hand_reading):
        stable_label = prediction.stable_label
        if hand_reading is None:
            return stable_label
        # Three-finger poses split into TWO distinct labels: tight
        # fingers -> "three_together" (chrome_mode_toggle's default
        # binding), spread fingers -> plain "three" (no global
        # action by default, consumed by YT router for skip-ad while
        # YT mode is forced). The static recognizer doesn't have
        # either in its base label set (only fist/one/two/four/mute/
        # wheel_pose/pinch), so a three-finger pose lands as
        # "neutral" -- we promote here.
        if stable_label in {"three", "neutral"}:
            if self._is_three_together(hand_reading):
                return "three_together"
            if self._is_three_apart(hand_reading):
                return "three"
        if stable_label in {"four", "neutral"} and self._is_four_together(hand_reading):
            return "four_together"
        return stable_label

    def _is_three_together(self, hand_reading) -> bool:
        fingers = hand_reading.fingers
        return (
            self._is_chrome_open_finger(fingers["index"])
            and self._is_chrome_open_finger(fingers["middle"])
            and self._is_chrome_open_finger(fingers["ring"])
            and self._is_folded_finger(fingers["thumb"], allow_partial=True)
            and self._is_folded_finger(fingers["pinky"])
            and self._spread_is_together(hand_reading, "index_middle", max_distance=0.38, min_strength=0.56)
            and self._spread_is_together(hand_reading, "middle_ring", max_distance=0.38, min_strength=0.54)
        )

    def _is_left_three_mrp(self, hand_reading) -> bool:
        """Detects the Discord-mode activator: left hand showing
        middle + ring + pinky extended, with index folded and thumb
        folded.

        The base recogniser labels this as "three" (it just counts
        extended fingers) — which collides with mouse-mode's left-3
        toggle that expects index + middle + ring. Splitting them
        out by exact finger set gives us two disjoint left-hand
        three-finger gestures: standard `three` = mouse, MRP = Discord.
        """
        if hand_reading is None:
            return False
        fingers = getattr(hand_reading, "fingers", None)
        if fingers is None:
            return False
        try:
            return (
                self._is_chrome_open_finger(fingers["middle"])
                and self._is_chrome_open_finger(fingers["ring"])
                and self._is_chrome_open_finger(fingers["pinky"])
                and self._is_folded_finger(fingers["index"])
                and self._is_folded_finger(fingers["thumb"], allow_partial=True)
            )
        except Exception:
            return False

    def _is_three_apart(self, hand_reading) -> bool:
        fingers = hand_reading.fingers
        return (
            self._is_chrome_open_finger(fingers["index"])
            and self._is_chrome_open_finger(fingers["middle"])
            and self._is_chrome_open_finger(fingers["ring"])
            and self._is_folded_finger(fingers["thumb"], allow_partial=True)
            and self._is_folded_finger(fingers["pinky"])
            and self._spread_is_apart(hand_reading, "index_middle", min_distance=0.50, min_strength=0.56)
            and self._spread_is_apart(hand_reading, "middle_ring", min_distance=0.48, min_strength=0.54)
        )

    def _is_thumb_only_extended(self, hand_reading) -> bool:
        fingers = hand_reading.fingers
        return (
            self._is_chrome_open_finger(fingers["thumb"])
            and self._is_folded_finger(fingers["index"])
            and self._is_folded_finger(fingers["middle"])
            and self._is_folded_finger(fingers["ring"])
            and self._is_folded_finger(fingers["pinky"])
        )

    def _is_thumb_up(self, hand_reading) -> bool:
        if not self._is_thumb_only_extended(hand_reading):
            return False
        try:
            wrist_y = float(hand_reading.landmarks[0][1])
            thumb_tip_y = float(hand_reading.landmarks[4][1])
        except Exception:
            return False
        # MediaPipe landmark y grows downward, so thumb-tip ABOVE the
        # wrist means thumb_tip_y < wrist_y. Threshold (0.06) guards
        # against a sideways thumb being mis-classified — the tip has
        # to be a clear chunk above the wrist, not just slightly above.
        return (wrist_y - thumb_tip_y) > 0.06

    def _is_thumb_down(self, hand_reading) -> bool:
        if not self._is_thumb_only_extended(hand_reading):
            return False
        try:
            wrist_y = float(hand_reading.landmarks[0][1])
            thumb_tip_y = float(hand_reading.landmarks[4][1])
        except Exception:
            return False
        return (thumb_tip_y - wrist_y) > 0.06

    def _is_four_together(self, hand_reading) -> bool:
        fingers = hand_reading.fingers
        return (
            all(self._is_chrome_open_finger(fingers[name]) for name in ("index", "middle", "ring", "pinky"))
            and self._is_folded_finger(fingers["thumb"], allow_partial=True)
            and self._spread_is_together(hand_reading, "index_middle", max_distance=0.40, min_strength=0.52)
            and self._spread_is_together(hand_reading, "middle_ring", max_distance=0.40, min_strength=0.52)
            and self._spread_is_together(hand_reading, "ring_pinky", max_distance=0.40, min_strength=0.52)
        )

    def _is_chrome_open_finger(self, finger) -> bool:
        return finger.state == "fully_open" or (
            finger.state == "partially_curled"
            and finger.openness >= 0.70
            and finger.curl <= 0.46
        )

    def _is_folded_finger(self, finger, *, allow_partial: bool = False) -> bool:
        allowed = {"mostly_curled", "closed"}
        if allow_partial:
            allowed.add("partially_curled")
        return finger.state in allowed

    def _spread_is_together(self, hand_reading, spread_name: str, *, max_distance: float, min_strength: float) -> bool:
        spread = hand_reading.spreads[spread_name]
        return (
            spread.state == "together"
            or spread.distance <= max_distance
            or (spread.together_strength >= min_strength and spread.apart_strength <= 0.42)
        )

    def _spread_is_apart(self, hand_reading, spread_name: str, *, min_distance: float, min_strength: float) -> bool:
        spread = hand_reading.spreads[spread_name]
        return (
            spread.state == "apart"
            or spread.distance >= min_distance
            or (spread.apart_strength >= min_strength and spread.together_strength <= 0.42)
        )

    def _reset_spotify_wheel(self, *, clear_cooldown: bool = False) -> None:
        self._spotify_wheel_candidate = "neutral"
        self._spotify_wheel_candidate_since = 0.0
        self._spotify_wheel_visible = False
        self._spotify_wheel_anchor = None
        self._spotify_wheel_selected_key = None
        self._spotify_wheel_selected_since = 0.0
        self._spotify_wheel_pose_grace_until = 0.0
        self._spotify_wheel_cursor_offset = None
        if clear_cooldown:
            self._spotify_wheel_cooldown_until = 0.0
        if self.spotify_wheel_overlay.isVisible():
            self.spotify_wheel_overlay.hide_overlay()

    def _reset_chrome_wheel(self, *, clear_cooldown: bool = False) -> None:
        self._chrome_wheel_candidate = "neutral"
        self._chrome_wheel_candidate_since = 0.0
        self._chrome_wheel_visible = False
        self._chrome_wheel_anchor = None
        self._chrome_wheel_selected_key = None
        self._chrome_wheel_selected_since = 0.0
        self._chrome_wheel_pose_grace_until = 0.0
        self._chrome_wheel_cursor_offset = None
        if clear_cooldown:
            self._chrome_wheel_cooldown_until = 0.0
        if self.chrome_wheel_overlay.isVisible():
            self.chrome_wheel_overlay.hide_overlay()

    def _reset_youtube_wheel(self, *, clear_cooldown: bool = False) -> None:
        self._youtube_wheel_candidate = "neutral"
        self._youtube_wheel_candidate_since = 0.0
        self._youtube_wheel_visible = False
        self._youtube_wheel_anchor = None
        self._youtube_wheel_selected_key = None
        self._youtube_wheel_selected_since = 0.0
        self._youtube_wheel_pose_grace_until = 0.0
        self._youtube_wheel_cursor_offset = None
        if clear_cooldown:
            self._youtube_wheel_cooldown_until = 0.0
        if self.youtube_wheel_overlay.isVisible():
            self.youtube_wheel_overlay.hide_overlay()

    def _youtube_absence_tick(self, hand_visible: bool) -> None:
        """Auto-pause the playing YouTube tab when no hand is visible
        for `youtube_pause_when_user_leaves_seconds`. Resume on hand
        return — but only if we were the one who paused (don't fight
        a manual pause)."""
        if not bool(getattr(self.config, "youtube_pause_when_user_leaves", False)):
            self._youtube_absence_started_at = 0.0
            self._youtube_paused_for_absence = False
            return
        try:
            tab_present = bool(self.youtube_controller.has_youtube_tab())
        except Exception:
            tab_present = False
        if not tab_present:
            self._youtube_absence_started_at = 0.0
            self._youtube_paused_for_absence = False
            return
        now = time.time()
        if hand_visible:
            if self._youtube_paused_for_absence:
                try:
                    self.youtube_controller.toggle_playback()
                except Exception:
                    pass
                self._youtube_paused_for_absence = False
            self._youtube_absence_started_at = 0.0
            return
        if self._youtube_absence_started_at <= 0.0:
            self._youtube_absence_started_at = now
            return
        if self._youtube_paused_for_absence:
            return
        threshold = max(2, int(getattr(self.config, "youtube_pause_when_user_leaves_seconds", 6)))
        if (now - self._youtube_absence_started_at) >= threshold:
            try:
                self.youtube_controller.toggle_playback()
                self._youtube_paused_for_absence = True
            except Exception:
                pass

    def _youtube_auto_skip_tick(self) -> None:
        """Poll the focused YouTube tab for a Skip Ad button at 1 Hz
        and click it when the template match crosses the configured
        threshold. Restores prior foreground after the click so the
        skip is invisible to the user."""
        if not bool(getattr(self.config, "youtube_auto_skip_ads", False)):
            return
        now = time.time()
        if (now - self._youtube_auto_skip_last_tick) < 1.0:
            return
        self._youtube_auto_skip_last_tick = now
        try:
            self.youtube_controller.auto_skip_ads_tick()
        except Exception:
            pass

    def _youtube_wheel_items(self) -> tuple[tuple[str, str, float], ...]:
        labels = (
            ("fullscreen", "Fullscreen"),
            ("theater", "Theater"),
            ("mini_player", "Mini Player"),
            ("captions", "Captions"),
            ("like", "Like"),
            ("dislike", "Dislike"),
            ("share", "Share"),
            ("speed_down", "Slower"),
            ("speed_up", "Faster"),
        )
        slice_span = 360.0 / len(labels)
        return tuple(
            (key, label, (90.0 - index * slice_span) % 360.0)
            for index, (key, label) in enumerate(labels)
        )

    def _youtube_wheel_label(self, key: str) -> str:
        for item_key, label, _angle in self._youtube_wheel_items():
            if item_key == key:
                return label
        return key.replace("_", " ")

    def _youtube_wheel_pose_active(self, prediction) -> bool:
        if prediction is None:
            return False
        stable_label = str(getattr(prediction, "stable_label", "neutral") or "neutral")
        raw_label = str(getattr(prediction, "raw_label", "neutral") or "neutral")
        confidence = float(getattr(prediction, "confidence", 0.0) or 0.0)
        if stable_label == "wheel_pose":
            return True
        if raw_label == "wheel_pose" and confidence >= 0.48:
            return True
        return False

    def _update_youtube_wheel(self, prediction, hand_reading, now: float, *, active: bool) -> bool:
        youtube_active = self._youtube_mode_info in {"forced", "auto"}
        if not active or prediction is None or hand_reading is None or not youtube_active:
            if self._youtube_wheel_visible and now >= self._youtube_wheel_pose_grace_until:
                self._youtube_control_text = "youtube wheel closed"
                self._reset_youtube_wheel()
            else:
                self._youtube_wheel_candidate = "neutral"
                self._youtube_wheel_candidate_since = now
            return self._youtube_wheel_visible

        wheel_pose = self._youtube_wheel_pose_active(prediction)
        if self._youtube_wheel_visible:
            if wheel_pose:
                self._youtube_wheel_pose_grace_until = now + 0.25
                self._update_youtube_wheel_selection(hand_reading, now)
            elif now >= self._youtube_wheel_pose_grace_until:
                self._youtube_control_text = "youtube wheel closed"
                self._reset_youtube_wheel()
            return True

        if now < self._youtube_wheel_cooldown_until:
            if not wheel_pose:
                self._youtube_wheel_candidate = "neutral"
            return False

        if not wheel_pose:
            self._youtube_wheel_candidate = "neutral"
            self._youtube_wheel_candidate_since = now
            return False

        if self._youtube_wheel_candidate != "youtube_wheel_pose":
            self._youtube_wheel_candidate = "youtube_wheel_pose"
            self._youtube_wheel_candidate_since = now
            self._youtube_control_text = "hold youtube wheel pose"
            return True

        if now - self._youtube_wheel_candidate_since < 1.0:
            return True

        self._youtube_wheel_visible = True
        self._youtube_wheel_anchor = hand_reading.palm.center.copy()
        self._youtube_wheel_cursor_offset = (0.0, 0.0)
        self._youtube_wheel_selected_key = None
        self._youtube_wheel_selected_since = now
        self._youtube_wheel_pose_grace_until = now + 0.25
        self._youtube_control_text = "youtube wheel active"
        return True

    def _update_youtube_wheel_selection(self, hand_reading, now: float) -> None:
        if self._youtube_wheel_anchor is None:
            self._youtube_wheel_anchor = hand_reading.palm.center.copy()
        offset = (hand_reading.palm.center - self._youtube_wheel_anchor) / max(hand_reading.palm.scale, 1e-6)
        self._youtube_wheel_cursor_offset = (float(offset[0]), float(offset[1]))
        selection_key = self._wheel_selection_key(float(offset[0]), float(offset[1]), self._youtube_wheel_items())
        if selection_key is None:
            self._youtube_wheel_selected_key = None
            self._youtube_wheel_selected_since = now
            self._youtube_control_text = "youtube wheel active"
            return
        if selection_key != self._youtube_wheel_selected_key:
            self._youtube_wheel_selected_key = selection_key
            self._youtube_wheel_selected_since = now
            self._youtube_control_text = f"youtube wheel: {self._youtube_wheel_label(selection_key)}"
            return
        if now - self._youtube_wheel_selected_since < 1.0:
            return
        self._execute_youtube_wheel_action(selection_key, now)
        self._youtube_wheel_cooldown_until = now + 1.5
        self._reset_youtube_wheel()

    def _execute_youtube_wheel_action(self, key: str, now: float) -> None:
        controller = self.youtube_controller
        if key == "fullscreen":
            controller.toggle_fullscreen()
        elif key == "theater":
            controller.toggle_theater()
        elif key == "mini_player":
            controller.toggle_mini_player()
        elif key == "captions":
            controller.toggle_captions()
        elif key == "like":
            controller.like_video()
        elif key == "dislike":
            controller.dislike_video()
        elif key == "share":
            controller.share_video()
            self.mouse_tracker.force_enable_mode(now)
            self._mouse_mode_enabled = True
            self._mouse_control_text = "Mouse mode on"
            self.voice_status_overlay.show_info_hint("Mouse mode on - click the Share options", duration=3.0)
        elif key == "speed_down":
            controller.speed_down()
        elif key == "speed_up":
            controller.speed_up()
        else:
            return
        self._youtube_control_text = controller.message
        action_key = f"youtube_{key}"
        if key == "captions" and "no captions available" in controller.message.lower():
            action_key = "youtube_captions_unavailable"
            self.voice_status_overlay.show_info_hint("No captions available for this video", duration=3.0)
        self.command_detected.emit(self._youtube_control_text)
        self._record_action(action_key, self._youtube_control_text)

    def _spotify_wheel_items(self) -> tuple[tuple[str, str, float], ...]:
        labels = (
            ("add_playlist", "Add Playlist"),
            ("remove_playlist", "Remove Playlist"),
            ("create_playlist", "Create Playlist"),
            ("add_queue", "Add Queue"),
            ("remove_queue", "Remove Queue"),
            ("like", "Add to Liked"),
            ("remove_liked", "Remove from Liked"),
            ("shuffle", "Shuffle"),
        )
        slice_span = 360.0 / len(labels)
        return tuple(
            (key, label, (90.0 - index * slice_span) % 360.0)
            for index, (key, label) in enumerate(labels)
        )

    def _chrome_wheel_items(self) -> tuple[tuple[str, str, float], ...]:
        return (
            ("bookmark", "Bookmark This Tab", 90.0),
            ("history", "History", 30.0),
            ("downloads", "Downloads", 330.0),
            ("bookmarks", "Bookmarks Manager", 270.0),
            ("print", "Print Page", 210.0),
            ("reopen", "Reopen Tab", 150.0),
        )

    def _chrome_active_for_wheel(self, now: float) -> bool:
        if now < self._chrome_active_cache_until:
            return self._chrome_active_cache
        active = bool(self.chrome_controller.is_window_active())
        self._chrome_active_cache = active
        self._chrome_active_cache_until = now + 0.55
        return active

    def _spotify_active_for_wheel(self, now: float) -> bool:
        if now < self._spotify_active_cache_until:
            return self._spotify_active_cache
        # Cache miss. is_active_for_wheel() may HTTP-probe the
        # Spotify Web API when no desktop window is visible â€” that
        # call blocks for up to 5 s while Spotify is still launching.
        # Run the refresh on a background thread and return the
        # previous cached value; the bg thread updates the cache and
        # the next tick reads the fresh result. Push the cache fence
        # forward so we don't fire a second probe per tick while the
        # first is still in flight.
        self._spotify_active_cache_until = now + 0.9
        if not self._spotify_active_refresh_in_flight:
            self._spotify_active_refresh_in_flight = True

            def _refresh() -> None:
                try:
                    if hasattr(self.spotify_controller, "is_active_for_wheel"):
                        active = bool(self.spotify_controller.is_active_for_wheel())
                    else:
                        active = bool(self.spotify_controller.is_active_device_available())
                    self._spotify_active_cache = active
                except Exception:
                    pass
                finally:
                    self._spotify_active_refresh_in_flight = False

            try:
                threading.Thread(
                    target=_refresh,
                    name="spotify-wheel-active-probe",
                    daemon=True,
                ).start()
            except Exception:
                self._spotify_active_refresh_in_flight = False
        return self._spotify_active_cache

    def _chrome_wheel_pose_active(self, prediction, now: float) -> bool:
        if prediction is None:
            return False
        if self._youtube_mode_info in {"forced", "auto"}:
            return False
        stable_label = str(getattr(prediction, "stable_label", "neutral") or "neutral")
        raw_label = str(getattr(prediction, "raw_label", "neutral") or "neutral")
        confidence = float(getattr(prediction, "confidence", 0.0) or 0.0)
        if stable_label == "chrome_wheel_pose":
            return True
        if raw_label == "chrome_wheel_pose" and confidence >= 0.50:
            return True
        if self._chrome_mode_enabled and self._chrome_active_for_wheel(now):
            if stable_label == "wheel_pose":
                return True
            if raw_label == "wheel_pose" and confidence >= 0.48:
                return True
        return False

    def _update_chrome_wheel(self, prediction, hand_reading, now: float, *, active: bool) -> bool:
        if not active or prediction is None or hand_reading is None:
            if self._chrome_wheel_visible and now >= self._chrome_wheel_pose_grace_until:
                self._chrome_control_text = "chrome wheel closed"
                self._reset_chrome_wheel()
            else:
                self._chrome_wheel_candidate = "neutral"
                self._chrome_wheel_candidate_since = now
            return self._chrome_wheel_visible

        wheel_label = self._chrome_wheel_pose_active(prediction, now)
        if self._chrome_wheel_visible:
            if wheel_label:
                self._chrome_wheel_pose_grace_until = now + 0.25
                self._update_chrome_wheel_selection(hand_reading, now)
            elif now >= self._chrome_wheel_pose_grace_until:
                self._chrome_control_text = "chrome wheel closed"
                self._reset_chrome_wheel()
            return True

        if now < self._chrome_wheel_cooldown_until:
            if not wheel_label:
                self._chrome_wheel_candidate = "neutral"
            return False

        if not wheel_label:
            self._chrome_wheel_candidate = "neutral"
            self._chrome_wheel_candidate_since = now
            return False

        if not self._chrome_active_for_wheel(now):
            self._chrome_control_text = "chrome must be active for wheel"
            self._chrome_wheel_candidate = "neutral"
            self._chrome_wheel_candidate_since = now
            return True

        if self._chrome_wheel_candidate != "chrome_wheel_pose":
            self._chrome_wheel_candidate = "chrome_wheel_pose"
            self._chrome_wheel_candidate_since = now
            self._chrome_control_text = "hold chrome wheel pose"
            return True

        if now - self._chrome_wheel_candidate_since < 1.0:
            return True

        self._chrome_wheel_visible = True
        self._chrome_wheel_anchor = hand_reading.palm.center.copy()
        self._chrome_wheel_cursor_offset = (0.0, 0.0)
        self._chrome_wheel_selected_key = None
        self._chrome_wheel_selected_since = now
        self._chrome_wheel_pose_grace_until = now + 0.25
        self._chrome_control_text = "chrome wheel active"
        return True

    def _update_chrome_wheel_selection(self, hand_reading, now: float) -> None:
        if self._chrome_wheel_anchor is None:
            self._chrome_wheel_anchor = hand_reading.palm.center.copy()
        offset = (hand_reading.palm.center - self._chrome_wheel_anchor) / max(hand_reading.palm.scale, 1e-6)
        self._chrome_wheel_cursor_offset = (float(offset[0]), float(offset[1]))
        selection_key = self._wheel_selection_key(float(offset[0]), float(offset[1]), self._chrome_wheel_items())
        if selection_key is None:
            self._chrome_wheel_selected_key = None
            self._chrome_wheel_selected_since = now
            self._chrome_control_text = "chrome wheel active"
            return
        if selection_key != self._chrome_wheel_selected_key:
            self._chrome_wheel_selected_key = selection_key
            self._chrome_wheel_selected_since = now
            self._chrome_control_text = f"chrome wheel: {self._chrome_wheel_label(selection_key)}"
            return
        if now - self._chrome_wheel_selected_since < 1.0:
            return
        self._execute_chrome_wheel_action(selection_key)
        self._chrome_wheel_cooldown_until = now + 1.5
        self._reset_chrome_wheel()

    def _wheel_selection_key(self, dx: float, dy: float, items: tuple[tuple[str, str, float], ...]) -> str | None:
        radius = math.hypot(dx, dy)
        # No upper bound on radius: as long as the cursor is past the
        # central deadzone, the angle alone determines the selected
        # slice. Previously a `radius > 1.25` cap dropped selection
        # the moment the user's hand drifted further from the wheel
        # center, which felt like "the cursor is clearly inside this
        # slice but nothing's selected".
        if radius < 0.59:
            return None
        angle = (math.degrees(math.atan2(-dy, dx)) + 360.0) % 360.0
        slice_span = 360.0 / max(1, len(items))
        half_slice = slice_span / 2.0
        angular_margin = 1.5
        for key, _label, target_angle in items:
            delta = abs((angle - target_angle + 180.0) % 360.0 - 180.0)
            if delta <= max(0.0, half_slice - angular_margin):
                return key
        return None

    def _chrome_wheel_label(self, key: str) -> str:
        for item_key, label, _angle in self._chrome_wheel_items():
            if item_key == key:
                return label
        return key.replace("_", " ")

    def _execute_chrome_wheel_action(self, key: str) -> None:
        if key == "bookmark":
            success = self.chrome_controller.bookmark_current_tab()
        elif key == "history":
            success = self.chrome_controller.open_history()
        elif key == "downloads":
            success = self.chrome_controller.open_downloads()
        elif key == "bookmarks":
            success = self.chrome_controller.open_bookmarks_manager()
        elif key == "print":
            success = self.chrome_controller.print_page()
        else:
            success = self.chrome_controller.reopen_closed_tab()
        self._chrome_control_text = self.chrome_controller.message
        self.command_detected.emit(self._chrome_control_text)
        if success:
            self._chrome_mode_enabled = True

    def _update_spotify_wheel(self, prediction, hand_reading, now: float, *, active: bool) -> bool:
        if not active or prediction is None or hand_reading is None:
            if self._spotify_wheel_visible and now >= self._spotify_wheel_pose_grace_until:
                self._spotify_control_text = "spotify wheel closed"
                self._reset_spotify_wheel()
            else:
                self._spotify_wheel_candidate = "neutral"
                self._spotify_wheel_candidate_since = now
            return self._spotify_wheel_visible

        wheel_label = prediction.stable_label == "wheel_pose" or (
            prediction.raw_label == "wheel_pose" and prediction.confidence >= 0.56
        )
        if self._spotify_wheel_visible:
            if wheel_label:
                self._spotify_wheel_pose_grace_until = now + 0.25
                self._update_spotify_wheel_selection(hand_reading, now)
            elif now >= self._spotify_wheel_pose_grace_until:
                self._spotify_control_text = "spotify wheel closed"
                self._reset_spotify_wheel()
            return True

        if now < self._spotify_wheel_cooldown_until:
            if not wheel_label:
                self._spotify_wheel_candidate = "neutral"
            return False

        if self._chrome_mode_enabled and self._chrome_active_for_wheel(now):
            self._spotify_wheel_candidate = "neutral"
            self._spotify_wheel_candidate_since = now
            return False

        if not wheel_label:
            self._spotify_wheel_candidate = "neutral"
            self._spotify_wheel_candidate_since = now
            return False

        if not self._spotify_active_for_wheel(now):
            self._spotify_control_text = self.spotify_controller.message or "spotify inactive on device"
            self._spotify_wheel_candidate = "neutral"
            self._spotify_wheel_candidate_since = now
            return True

        if self._spotify_wheel_candidate != "wheel_pose":
            self._spotify_wheel_candidate = "wheel_pose"
            self._spotify_wheel_candidate_since = now
            self._spotify_control_text = "hold wheel pose for spotify wheel"
            return True

        if now - self._spotify_wheel_candidate_since < 1.0:
            return True

        self._spotify_wheel_visible = True
        self._spotify_wheel_anchor = hand_reading.palm.center.copy()
        self._spotify_wheel_cursor_offset = (0.0, 0.0)
        self._spotify_wheel_selected_key = None
        self._spotify_wheel_selected_since = now
        self._spotify_wheel_pose_grace_until = now + 0.25
        self._spotify_control_text = "spotify wheel active"
        return True

    def _update_spotify_wheel_selection(self, hand_reading, now: float) -> None:
        if self._spotify_wheel_anchor is None:
            self._spotify_wheel_anchor = hand_reading.palm.center.copy()
        offset = (hand_reading.palm.center - self._spotify_wheel_anchor) / max(hand_reading.palm.scale, 1e-6)
        self._spotify_wheel_cursor_offset = (float(offset[0]), float(offset[1]))
        selection_key = self._spotify_wheel_selection_key(float(offset[0]), float(offset[1]))
        if selection_key is None:
            self._spotify_wheel_selected_key = None
            self._spotify_wheel_selected_since = now
            self._spotify_control_text = "spotify wheel active"
            return
        if selection_key != self._spotify_wheel_selected_key:
            self._spotify_wheel_selected_key = selection_key
            self._spotify_wheel_selected_since = now
            self._spotify_control_text = f"spotify wheel: {self._spotify_wheel_label(selection_key)}"
            return
        if now - self._spotify_wheel_selected_since < 1.0:
            return
        self._execute_spotify_wheel_action(selection_key)
        self._spotify_wheel_cooldown_until = now + 1.5
        self._reset_spotify_wheel()

    def _spotify_wheel_selection_key(self, dx: float, dy: float) -> str | None:
        return self._wheel_selection_key(dx, dy, self._spotify_wheel_items())

    def _spotify_wheel_label(self, key: str) -> str:
        for item_key, label, _angle in self._spotify_wheel_items():
            if item_key == key:
                return label
        return key.replace("_", " ")

    def _execute_spotify_wheel_action(self, key: str) -> None:
        if key == "add_playlist":
            self._spotify_control_text = "say what playlist you would like to add to"
            self.command_detected.emit(self._spotify_control_text)
            self._start_playlist_prompt("add_playlist")
            return
        if key == "create_playlist":
            self._spotify_control_text = "what would you like to call the playlist?"
            self.command_detected.emit(self._spotify_control_text)
            self._start_playlist_prompt("create_playlist")
            return
        if key == "remove_playlist":
            success = self.spotify_controller.remove_current_track_from_current_playlist()
            self._spotify_control_text = self.spotify_controller.message
            self.command_detected.emit(self._spotify_control_text)
            if success:
                details = self.spotify_controller.get_current_track_details()
                if details is not None:
                    self._spotify_info_text = details.summary()
            return
        if key == "add_queue":
            success = self.spotify_controller.add_current_track_to_queue()
        elif key == "remove_queue":
            success = self.spotify_controller.remove_current_track_from_queue()
        elif key == "remove_liked":
            success = self.spotify_controller.remove_current_track_from_liked()
        elif key == "shuffle":
            success = self.spotify_controller.toggle_shuffle()
        else:
            success = self.spotify_controller.save_current_track()
        self._spotify_control_text = self.spotify_controller.message
        self.command_detected.emit(self._spotify_control_text)
        if success:
            details = self.spotify_controller.get_current_track_details()
            if details is not None:
                self._spotify_info_text = details.summary()

    def _start_playlist_prompt(self, action: str) -> None:
        if self._voice_listening:
            return
        self._voice_cooldown_until = time.monotonic() + 0.7
        self._start_voice_capture(mode=action, preferred_app=None)

    def _clean_playlist_reply(self, spoken_text: str) -> str:
        cleaned = str(spoken_text or "").lower()
        cleaned = cleaned.replace("feel good", "feel-good")
        cleaned = re.sub(
            r"\b(and|then|uh|um|add|remove|it|this|current|song|track|playlist|to|from|my|the|please|thanks|thank you|would like|like|called|named|titled)\b",
            " ",
            cleaned,
        )
        cleaned = re.sub(r"\s+", " ", cleaned).strip(" .-")
        return cleaned

    def _clean_create_playlist_reply(self, spoken_text: str) -> str:
        cleaned = str(spoken_text or "").strip()
        cleaned = re.sub(
            r"^\s*(please\s+)?(can you\s+)?(create|make|new|build|set up)\s+(a\s+)?(new\s+)?(playlist\s+)?(called|named|titled)?\s*",
            "",
            cleaned,
            flags=re.IGNORECASE,
        )
        cleaned = re.sub(
            r"\b(uh|um|please|thanks|thank you)\b",
            " ",
            cleaned,
            flags=re.IGNORECASE,
        )
        cleaned = re.sub(r"\s+", " ", cleaned).strip(" .-")
        words = [word.capitalize() if word.islower() else word for word in cleaned.split(" ") if word]
        return " ".join(words)

    def _handle_left_hand_voice(self, prediction, now: float) -> None:
        # Apply Gesture Binds remap for the left hand. Same semantics
        # as the right-hand call site in _handle_app_controls — see the
        # helper docstring for why this is the chokepoint.
        prediction = self._apply_gesture_binding_remap(prediction, "Left", now)
        stable_label = prediction.stable_label
        # r53: Left-fist cancel is restored. Dictation stays retired;
        # "two" is still instant-clip. Fist cancels an in-progress
        # voice command, dictation, save-location prompt, or file
        # picker. A canceled save prompt keeps the file in the default
        # folder (MainWindow treats canceled=True as action=default).
        trigger_labels = {"one"}

        if stable_label == "fist":
            voice_busy = (
                self._voice_listening
                or self._dictation_active
                or bool(getattr(self, "_save_prompt_active", False))
                or bool(getattr(self, "_selection_prompt_active", False))
            )
            if not voice_busy:
                self._reset_voice_candidate(now)
                if self._voice_latched_label == "fist":
                    self._voice_latched_label = None
                return
            if self._voice_candidate != "fist":
                self._voice_candidate = "fist"
                self._voice_candidate_since = now
                return
            if now < self._voice_cooldown_until:
                return
            if now - self._voice_candidate_since < 0.35:
                return
            self._voice_latched_label = "fist"
            self._voice_cooldown_until = now + 1.0
            self._cancel_all_voice_stages()
            return

        if self._voice_latched_label == "fist":
            self._voice_latched_label = None

        if stable_label == self._voice_latched_label:
            if stable_label not in trigger_labels:
                self._voice_latched_label = None
            return

        if stable_label not in trigger_labels:
            self._reset_voice_candidate(now)
            return

        if stable_label != self._voice_candidate:
            self._voice_candidate = stable_label
            self._voice_candidate_since = now
            self._begin_voice_hold_anchor()
            return

        if self._voice_one_hold_is_translating():
            # Hand is doing a swipe (or other committed move), not a
            # stationary listen pose. Restart the hold from here so a
            # later still "one" can still arm voice.
            self._reset_voice_candidate(now)
            return

        if stable_label == "one" and (self._voice_listening or self._dictation_active):
            return
        if now < self._voice_cooldown_until:
            return
        if now - self._voice_candidate_since < _STATIC_GESTURE_HOLD_SECONDS:
            return

        self._voice_latched_label = stable_label
        self._voice_cooldown_until = now + 1.25
        self._voice_one_two_triggered_at = now
        if self._selection_prompt_active:
            self._start_voice_capture(mode="selection", preferred_app=None)
            return
        self._start_voice_command()

    def _handle_left_hand_clip_gesture(self, prediction, hand_reading, now: float) -> None:
        """LEFT-hand TWO (V-shape) held for 1.0 s → instant clip.

        r53 v7: gesture swapped from left-fist to left-two. Dictation
        used to live on left-two but was retired; the fist-based clip
        also went away (fist now does nothing at all). Left-two hold
        is the single canonical gesture for the buffered clip trigger.

        After a clip triggers, a 3-s cooldown prevents the held
        pose from re-triggering.
        """
        if prediction is None or hand_reading is None:
            self._clip_gesture_candidate_since = 0.0
            return
        if now < self._clip_gesture_cooldown_until:
            return
        # Don't trigger while a voice command / save prompt / selection
        # prompt is up — clip would interrupt user's in-progress voice
        # interaction.
        if (self._voice_listening or self._dictation_active
                or self._save_prompt_active or self._selection_prompt_active):
            self._clip_gesture_candidate_since = 0.0
            return
        try:
            stable_label = str(getattr(prediction, "stable_label", "neutral") or "neutral")
        except Exception:
            self._clip_gesture_candidate_since = 0.0
            return
        if stable_label != "two":
            self._clip_gesture_candidate_since = 0.0
            return
        # First detection — start the hold timer.
        if self._clip_gesture_candidate_since <= 0.0:
            self._clip_gesture_candidate_since = now
            return
        # Hold satisfied? Threshold matches voice trigger hold
        # (1.0 s) for muscle-memory parity.
        if now - self._clip_gesture_candidate_since < _STATIC_GESTURE_HOLD_SECONDS:
            return
        # Trigger. Reset candidate and arm cooldown so the same
        # held pose doesn't fire again.
        self._clip_gesture_candidate_since = 0.0
        self._clip_gesture_cooldown_until = now + 3.0
        # Resolve duration from config (snap to supported set).
        try:
            duration_s = int(
                getattr(self.config, "clip_default_duration_seconds", 60) or 60
            )
        except Exception:
            duration_s = 60
        # Stash end_ts so main_window's dispatch can pin the clip
        # to the moment the gesture confirmed (= NOW). Mirrors the
        # voice "clip that" path at noop_engine.py:9474, which
        # stores result.speech_end_ts — a WALL-CLOCK timestamp
        # (time.time()). The caller's `now` is time.monotonic()
        # (seconds since boot), not wall-clock. Storing monotonic
        # here was the v1.1.4 fist-clip bug: main_window's export
        # computed `tail_to_drop = now_wall - end_ts` and got
        # ~56 YEARS, clamping trim_duration to 0 → single-frame
        # output. Always use time.time() for cross-clock anchors.
        try:
            import time as _cg_time
            self._pending_clip_voice_end_ts = float(_cg_time.time())
        except Exception:
            self._pending_clip_voice_end_ts = None
        # Toast + diag log so the user knows the gesture registered
        # before the export wall-time (~10-30 s for HQ encode).
        # r53: gated on Settings > General > Overlay > Text pop-ups.
        # Was bypassing the flag before, so users who turned pop-ups
        # off still saw the "Clipping last N seconds" pill on trigger.
        # label_text is also emitted via command_detected below, which
        # a dev log consumer might still want, so we build it either
        # way and only skip the visible overlay.
        try:
            mins = duration_s // 60
            if mins >= 2:
                label_text = f"Clipping last {mins} minutes"
            elif mins == 1:
                label_text = "Clipping last 1 minute"
            else:
                label_text = f"Clipping last {duration_s} seconds"
            try:
                _popups_on = bool(
                    getattr(self.config, "overlay_text_popups_enabled", True)
                )
            except Exception:
                _popups_on = True
            if _popups_on:
                self.voice_status_overlay.show_info_hint(label_text, duration=2.5)
        except Exception:
            pass
        try:
            self.command_detected.emit(label_text)
        except Exception:
            pass
        try:
            import sys as _cg_sys, time as _cg_time
            _cg_sys.stderr.write(
                f"[clip-gesture] left-two hold triggered at {now:.3f} "
                f"duration={duration_s}s -> queueing clip_gesture\n"
            )
            _cg_sys.stderr.flush()
        except Exception:
            pass
        try:
            self._queue_utility_request("clip_gesture")
        except Exception:
            pass
        try:
            self._record_action("clip_gesture", label_text)
        except Exception:
            pass

    def _begin_voice_hold_anchor(self) -> None:
        self._voice_hold_origin_xy = None
        self._voice_hold_origin_scale = 1.0
        xy = self._palm_xy(getattr(self, "_left_hand_reading", None))
        if xy is None:
            return
        try:
            scale = float(self._left_hand_reading.palm.scale)
        except Exception:
            scale = 1.0
        self._voice_hold_origin_xy = xy
        self._voice_hold_origin_scale = max(scale, 1e-6)

    def _voice_one_hold_is_translating(self) -> bool:
        """True when left-hand palm has moved swipe-scale since hold start.

        Small jitter and a little drift stay under the cap so a held
        "one" still starts voice. Missing landmarks fail open (no
        reading → not treated as moving).
        """
        reading = getattr(self, "_left_hand_reading", None)
        xy = self._palm_xy(reading)
        if xy is None:
            return False
        if self._voice_hold_origin_xy is None:
            self._begin_voice_hold_anchor()
            return False
        try:
            scale = float(reading.palm.scale)
        except Exception:
            scale = self._voice_hold_origin_scale
        scale = max(float(scale), float(self._voice_hold_origin_scale), 1e-6)
        return _palm_net_travel_exceeds(self._voice_hold_origin_xy, xy, scale)

    def _reset_voice_candidate(self, now: float) -> None:
        self._voice_candidate = "neutral"
        self._voice_candidate_since = now
        self._voice_hold_origin_xy = None
        self._voice_hold_origin_scale = 1.0
        if self._voice_latched_label is not None:
            self._voice_latched_label = None

    def _voice_mode_text(self) -> str:
        if self._dictation_active:
            return "dictation"
        if self._voice_listening:
            if self._voice_mode == "general":
                return "command"
            return self._voice_mode.replace("_", " ")
        return "ready"

    def _voice_preview_text(self, text: str, *, max_chars: int = 220) -> str:
        value = str(text or "").strip()
        if not value:
            return "-"
        if len(value) <= max_chars:
            return value
        return "..." + value[-(max_chars - 3):]

    # Verbs in a parsed voice intent that map to "the user is asking
    # us to OPEN something" (vs. play music, search, etc). Headline
    # text for these flips from generic "Executing command" to
    # "Launching <pretty app name>" so users see specifically what
    # is being launched instead of a generic acknowledgment.
    _LAUNCH_INTENT_ACTIONS = {
        "open", "launch", "start", "run", "show", "boot",
        "fire_up", "boot_up", "pull_up", "bring_up", "switch_to",
        "focus", "go_to", "navigate", "navigate_to", "load",
    }
    # Pretty-display map for known app keys. Anything not in here
    # falls through to a Title-Cased version of the key (e.g.
    # "spotify" â†’ "Spotify"). Keep these keys aligned with the
    # APP_ALIASES dict in voice/command_processor.py.
    _LAUNCH_APP_FRIENDLY = {
        "spotify": "Spotify",
        "chrome": "Chrome",
        "edge": "Edge",
        "firefox": "Firefox",
        "youtube": "YouTube",
        "discord": "Discord",
        "steam": "Steam",
        "outlook": "Outlook",
        "settings": "Settings",
        "file_explorer": "File Explorer",
        "files": "File Explorer",
        "notepad": "Notepad",
        "word": "Word",
        "excel": "Excel",
        "powerpoint": "PowerPoint",
        "visual_studio": "Visual Studio",
        "visual_studio_code": "VS Code",
        "github": "GitHub",
        "gmail": "Gmail",
        "chatgpt": "ChatGPT",
        "reddit": "Reddit",
    }

    def _build_launch_overlay_headline(self, payload: dict) -> str:
        """Pick the headline shown in the voice-status overlay when a
        voice command succeeds. For open/launch-style intents on a
        known app, returns 'Launching Spotify' / 'Launching Chrome'
        etc. Falls back to the generic 'Executing command' for
        actions like 'play <song>' or commands without a clean
        app intent."""
        try:
            action = str(payload.get("intent_action") or "").strip().lower()
            app_name = str(payload.get("intent_app_name") or "").strip().lower()
        except Exception:
            return "Executing command"
        if not app_name or not action:
            return "Executing command"
        if action not in self._LAUNCH_INTENT_ACTIONS:
            return "Executing command"
        friendly = self._LAUNCH_APP_FRIENDLY.get(app_name)
        if friendly is None:
            friendly = app_name.replace("_", " ").title()
        return f"Launching {friendly}"

    def _start_voice_command(self) -> None:
        chrome_mode_voice = self._chrome_mode_enabled and self.chrome_controller.is_window_open()
        self._start_voice_capture(mode="general", preferred_app="chrome" if chrome_mode_voice else None)
        self._record_action("voice_command_listen", "voice listening started")

    def _start_dictation_capture(self) -> None:
        if self._voice_listening or self._dictation_active:
            return
        # Latch the external (non-HGR) window the user had focused before
        # making the gesture -- that's where dictated text must land. Must
        # happen BEFORE any overlay shows, because the overlay may pull
        # foreground briefly.
        try:
            self.text_input_controller.capture_target_window()
        except Exception:
            pass
        # Short toggle-off rearm so a second "two" stops dictation
        # without feeling laggy. The release-required gate + candidate
        # hold still prevent instant accidental re-toggles.
        self._dictation_toggle_release_required = True
        self._dictation_release_candidate_since = 0.0
        self._dictation_stop_rearm_at = time.monotonic() + 0.9
        # Route through the proven voice-capture pipeline: whisper.cpp
        # (or faster-whisper) -> text_input_controller.insert_text.
        # That path is what the rest of the app's voice features use
        # and is known to actually type into the target window.
        self._start_voice_capture(mode="dictation")
        self._record_action("dictation_start", "dictation started")

    def _stop_dictation_capture(self) -> None:
        if not self._dictation_active:
            return
        # Signal the dictation worker to break out of its listen loop.
        if self._voice_stop_event is not None:
            try:
                self._voice_stop_event.set()
            except Exception:
                pass
        self._dictation_active = False
        self._dictation_toggle_release_required = False
        self._dictation_release_candidate_since = 0.0
        self._dictation_stop_rearm_at = 0.0
        self._voice_listening = False
        self._dictation_backend = "idle"
        self._voice_mode = "ready"
        self._voice_control_text = "dictation stopped"
        self._record_action("dictation_stop", "dictation stopped")
        # Flush any pending buffered final before tearing state down.
        try:
            flush = self._dictation_state.get("_flush_pending") if self._dictation_state else None
            if callable(flush):
                flush()
        except Exception:
            pass
        try:
            self.grammar_corrector.stop()
        except Exception:
            pass
        try:
            if self.whisper_refiner is not None:
                self.whisper_refiner.stop()
        except Exception:
            pass
        self._dictation_state = None
        # Hide the mic indicator silently â€” no result popup.
        try:
            self.voice_status_overlay.hide_overlay()
        except Exception:
            pass

    def _start_voice_capture(self, *, mode: str, preferred_app: str | None = None) -> None:
        if self._voice_listening:
            return
        self._voice_listening = True
        self._voice_mode = mode
        self._voice_request_id += 1
        request_id = self._voice_request_id
        self._voice_stop_event = threading.Event() if mode == "dictation" else None
        self._dictation_active = mode == "dictation"
        self._dictation_backend = "whisper_loop" if mode == "dictation" else "idle"
        if mode == "dictation":
            self._voice_control_text = "dictation listening..."
        elif mode == "save_prompt":
            self._voice_control_text = self._save_prompt_text
        else:
            self._voice_control_text = "voice listening..."
        self._voice_heard_text = "-"
        self._voice_display_text = "-"

        if mode == "dictation":
            stop_event = self._voice_stop_event
            self.grammar_corrector.set_callback(self._apply_grammar_correction)
            self.grammar_corrector.start()
            # The refiner and streaming hypotheses are MUTUALLY EXCLUSIVE: both
            # re-type the just-finalized text, and the refiner opens a SECOND
            # mic InputStream (two simultaneous captures on one device = dropped
            # frames). With live hypotheses on, the high-quality final decode +
            # grammar pass already cover what the refiner did, so we keep it off
            # to avoid the race/duplication and the extra mic stream.
            if self.whisper_refiner is not None and not _DICTATION_HYP_ENABLED:
                try:
                    self.whisper_refiner.set_callback(self._apply_refinement)
                    started = self.whisper_refiner.start()
                    print(f"[hgr] whisper_refiner.start -> {started} ({self.whisper_refiner.message})")
                except Exception as exc:
                    print(f"[hgr] whisper_refiner start failed: {exc}")
            elif self.whisper_refiner is not None:
                print("[hgr] whisper_refiner: skipped (streaming hypotheses enabled)")
            self.voice_status_overlay.show_processing("Preparing Dictation Mode")
            self.command_detected.emit(self._voice_control_text)
            self._emit_status("dictation active")

            def _dictation_worker() -> None:
                final_message = "dictation stopped"
                hold_back = 2
                # Trailing-latency knob. With live hypotheses now typing the
                # bulk of each utterance as it is spoken, the final only needs
                # to land fast enough to reconcile the held-back tail; it no
                # longer has to be the primary latency buffer. Lowered 2.5 -> 1.0.
                # (A new utterance's first hypothesis also flushes any still-
                # pending previous final immediately, so cross-utterance
                # reconciliation does not depend on this window.)
                _PENDING_WINDOW_SECONDS = 1.0
                state = {
                    "committed": "",
                    "last_words": [],
                    "final_display": self.dictation_processor.full_text,
                    "last_final": "",
                    "recent_finals": [],
                    "last_final_text": "",
                    "last_final_typed_len": 0,
                    "last_final_time": 0.0,
                    "stream_hyp_chars": 0,
                    "last_final_refined": False,
                    "pending_final_text": None,
                    "pending_final_timer": None,
                    "last_hypothesis_time": 0.0,
                    # Last char physically typed into the target window. Drives the
                    # utterance-seam leading-space decision in _seam_normalize so a
                    # dropped boundary space ("ofFree") is filled in without ever
                    # doubling a space in the normal (trailing-space-present) case.
                    "last_char": "",
                }
                pending_lock = threading.Lock()
                self._dictation_state = state

                def _common_prefix(a: list[str], b: list[str]) -> list[str]:
                    out: list[str] = []
                    for x, y in zip(a, b):
                        if x == y:
                            out.append(x)
                        else:
                            break
                    return out

                def _commit_final(text: str) -> None:
                    normalized = " ".join(text.lower().split())
                    now = time.monotonic()
                    payload = None
                    # The ENTIRE body runs under _corrector_lock so every read and
                    # write of the shared char-bookkeeping (committed,
                    # stream_hyp_chars, last_final_typed_len, recent_finals) is
                    # atomic against a concurrent hypothesis on the stream thread.
                    # _commit_final is only ever called WITHOUT the lock held
                    # (the pending Timer, or _flush_pending from the stream/command/
                    # stop paths), so this single acquisition is safe for the
                    # non-reentrant lock — no nested re-acquisition anywhere below.
                    with self._corrector_lock:
                        prev_final_text = state.get("last_final_text", "")
                        last_final_time = state.get("last_final_time", 0.0)
                        time_gap = (now - last_final_time) if last_final_time > 0 else 999.0
                        no_new_audio = last_final_time > 0 and time_gap < 2.0
                        if (
                            normalized
                            and normalized in state["recent_finals"]
                            and no_new_audio
                        ):
                            print(f"[dictation] commit: dropped re-emission (gap={time_gap:.1f}s) {text!r}")
                            state["committed"] = ""
                            state["last_words"] = []
                            state["stream_hyp_chars"] = 0
                            return
                        redecode_done = False
                        overlap_hit = bool(prev_final_text) and _redecode_overlap(prev_final_text, text)
                        print(f"[dictation] commit: text={text!r} prev={prev_final_text!r} gap={time_gap:.2f}s overlap={overlap_hit}")
                        # Cross-utterance re-decode/overlap revert: only valid when
                        # NO live hypotheses typed the current utterance (its revert
                        # count assumes the previous final is the only thing on
                        # screen). If hypotheses live-typed this utterance, the
                        # within-utterance reconciliation below owns the seam.
                        if (
                            prev_final_text
                            and time_gap < 10.0
                            and overlap_hit
                            and state.get("stream_hyp_chars", 0) == 0
                        ):
                            # stream_hyp_chars is 0 here (gate above), so the revert
                            # count is exactly the previous final's typed length.
                            revert_chars = state.get("last_final_typed_len", 0)
                            if 0 < revert_chars <= 300:
                                to_type = text + " "
                                removed = self.text_input_controller.remove_text(revert_chars)
                                if removed:
                                    inserted = self.text_input_controller.insert_text(to_type, prefer_paste=False)
                                    if inserted:
                                        self.grammar_corrector.sync_replace(revert_chars, to_type)
                                        print(f"[dictation] re-decode replace: -{revert_chars} +{len(to_type)} chars")
                                        redecode_done = True

                        if redecode_done:
                            current_display = state["final_display"] or ""
                            prev_norm = " ".join(prev_final_text.lower().split())
                            if prev_norm and current_display.lower().endswith(prev_norm):
                                current_display = current_display[: len(current_display) - len(prev_final_text)].rstrip()
                            state["final_display"] = (current_display + " " + text).strip() if current_display else text
                            state["last_final_text"] = text
                            state["last_final_typed_len"] = len(text + " ")
                            state["last_final_time"] = now
                            state["last_final_refined"] = False
                            state["stream_hyp_chars"] = 0
                            state["committed"] = ""
                            state["last_words"] = []
                            state["last_char"] = " "  # to_type ended with a space
                            state["last_final"] = normalized
                            if state["recent_finals"]:
                                state["recent_finals"][-1] = normalized
                            else:
                                state["recent_finals"].append(normalized)
                        else:
                            # --- reconcile the high-accuracy final against the text
                            # live hypotheses already typed for THIS utterance -----
                            # `committed` is the running prefix the hypothesis path
                            # typed (no trailing space); `committed_chars` (==
                            # stream_hyp_chars) is exactly how many chars landed on
                            # screen. We compare by WORD, case/punctuation-
                            # insensitively: greedy partials and the beam final often
                            # differ in casing/punctuation, and a char-exact
                            # `startswith` would treat the whole final as new and
                            # RE-TYPE it on top of the live text (duplication).
                            # Instead: keep the agreed prefix, backspace any
                            # divergent/over-long live tail, type only the final's
                            # clean remainder.
                            committed = state["committed"]
                            committed_chars = state.get("stream_hyp_chars", 0)
                            backspace_chars, to_type, common_chars = _reconcile_final_edit(
                                committed, committed_chars, text
                            )
                            lead = ""
                            if committed == "":
                                # Commit-only path (no live hypotheses typed this
                                # utterance) -> this is the utterance's first content,
                                # so fix the seam: re-case the leading word and fill a
                                # dropped boundary space. to_type was "text " (k==0).
                                # `lead` is a separator and is NOT counted in
                                # typed_chars / last_final_typed_len.
                                lead, cased = _seam_normalize(
                                    text, state.get("final_display", ""), state.get("last_char", "")
                                )
                                to_type = cased + " "
                            ok = True
                            if backspace_chars > 0:
                                # divergent or over-long live tail -> remove it so the
                                # authoritative final wins the seam (no duplication).
                                ok = self.text_input_controller.remove_text(backspace_chars)
                                if ok:
                                    self.grammar_corrector.sync_replace(backspace_chars, "")
                            typed_chars = 0
                            if ok:
                                # Commit-only path keeps clipboard PASTE unchanged;
                                # streaming remainders are short and use SendInput so
                                # we never clobber the clipboard.
                                use_paste = committed == ""
                                inserted = self.text_input_controller.insert_text(lead + to_type, prefer_paste=use_paste)
                                if not inserted and backspace_chars > 0:
                                    # We already removed the live tail; retry the
                                    # insert once so a transient focus blip between
                                    # the backspace and the insert can't drop the word.
                                    inserted = self.text_input_controller.insert_text(lead + to_type, prefer_paste=use_paste)
                                if inserted:
                                    self.grammar_corrector.append(lead + to_type)
                                    typed_chars = len(to_type)
                                    state["last_char"] = to_type[-1] if to_type else state.get("last_char", "")
                                else:
                                    ok = False
                            current_display = state["final_display"] or ""
                            state["final_display"] = (current_display + " " + text).strip() if current_display else text
                            state["last_final_text"] = text
                            # Only count chars that actually landed. If focus was lost
                            # mid-commit, keep the typed-len at a safe value so a later
                            # refinement/re-decode never backspaces phantom characters.
                            state["last_final_typed_len"] = (common_chars + typed_chars) if ok else 0
                            state["last_final_time"] = now
                            state["last_final_refined"] = False
                            state["stream_hyp_chars"] = 0
                            state["committed"] = ""
                            state["last_words"] = []
                            state["last_final"] = normalized
                            state["recent_finals"].append(normalized)
                            if len(state["recent_finals"]) > 4:
                                state["recent_finals"] = state["recent_finals"][-4:]
                        payload = {
                            "event": "dictation_chunk",
                            "success": True,
                            "heard_text": text,
                            "control_text": "dictation typing",
                            "display_text": state["final_display"],
                            "partial": False,
                        }
                    if payload is not None:
                        self._voice_queue.put((request_id, payload))

                def _flush_pending() -> None:
                    with pending_lock:
                        text = state.get("pending_final_text")
                        state["pending_final_text"] = None
                        state["pending_final_timer"] = None
                    if text:
                        try:
                            _commit_final(text)
                        except Exception as exc:
                            print(f"[dictation] _flush_pending error: {exc}")

                def _buffer_final(text: str) -> None:
                    with pending_lock:
                        pending = state.get("pending_final_text")
                        normalized_new = " ".join(text.lower().split())
                        # Drop whisper's buffered re-emission of a just-
                        # committed chunk. With commit-only decoding the
                        # streamer cannot replay old audio, so "re-emission"
                        # collapses to "two identical commits within a short
                        # window" â€” anything past that is a legitimate user
                        # repeat.
                        last_final_time = state.get("last_final_time", 0.0)
                        now_m = time.monotonic()
                        since_commit = (now_m - last_final_time) if last_final_time > 0 else 999.0
                        no_new_audio = last_final_time > 0 and since_commit < 2.0
                        if no_new_audio:
                            for prior in state.get("recent_finals", []):
                                if not prior:
                                    continue
                                if normalized_new == prior:
                                    print(f"[dictation] buffer: dropped exact re-emission (gap={since_commit:.1f}s, no new hyp) {text!r}")
                                    return
                                if (
                                    prior.endswith(normalized_new)
                                    and len(normalized_new) < len(prior)
                                ):
                                    print(f"[dictation] buffer: dropped suffix re-emission (gap={since_commit:.1f}s, no new hyp) {text!r}")
                                    return
                        # Exact-dup of current pending: ignore without resetting
                        # the timer. Whisper keeps emitting the same final during
                        # trailing silence; we don't want those to delay commit.
                        if pending:
                            normalized_pending = " ".join(pending.lower().split())
                            if normalized_new == normalized_pending:
                                return
                            # Suffix-dup: new is already contained at the end of
                            # pending. Happens after a glue when whisper re-emits
                            # just the last fragment. Drop silently.
                            if normalized_pending.endswith(normalized_new) and len(normalized_new) < len(normalized_pending):
                                return
                        prev_timer = state.get("pending_final_timer")
                        if prev_timer is not None:
                            prev_timer.cancel()
                        if pending:
                            merged = (pending + " " + text).strip()
                            print(f"[dictation] pending glue: {pending!r} + {text!r} -> {merged!r}")
                            state["pending_final_text"] = merged
                        else:
                            state["pending_final_text"] = text
                            print(f"[dictation] pending new: {text!r}")
                        timer = threading.Timer(_PENDING_WINDOW_SECONDS, _flush_pending)
                        timer.daemon = True
                        timer.start()
                        state["pending_final_timer"] = timer

                state["_flush_pending"] = _flush_pending

                def _terminate_live_utterance() -> None:
                    # The whole-utterance final decode stripped to nothing
                    # (trailing-silence hallucination / no-speech), but live
                    # hypotheses may already have typed LocalAgreement-stable words
                    # for this utterance. Keep that confirmed text — it was real
                    # speech — but terminate it with a separator space and RESTORE
                    # the bookkeeping invariant (stream_hyp_chars == len(committed)
                    # == 0). Without zeroing stream_hyp_chars here, the next
                    # utterance's reconciler would read a stale-high count and
                    # backspace into previously-committed text.
                    #
                    # First flush any still-pending PREVIOUS final: a very short
                    # utterance (too brief for partials) can reach here while the
                    # prior utterance's final is still buffered, in which case
                    # `committed` below would still hold the PRIOR utterance's live
                    # text. Committing it first (outside the lock; _flush_pending
                    # re-takes it) resets committed/stream_hyp_chars so we never
                    # steal it and then double-commit it when its timer fires.
                    if state.get("pending_final_text"):
                        _flush_pending()
                    with self._corrector_lock:
                        committed = state.get("committed", "")
                        hyp_chars = state.get("stream_hyp_chars", 0)
                        if hyp_chars > 0:
                            inserted = self.text_input_controller.insert_text(" ", prefer_paste=False)
                            if inserted:
                                self.grammar_corrector.append(" ")
                                state["last_char"] = " "
                            current_display = state["final_display"] or ""
                            state["final_display"] = (
                                (current_display + " " + committed).strip() if current_display else committed
                            )
                            state["last_final_text"] = committed
                            state["last_final_typed_len"] = hyp_chars + (1 if inserted else 0)
                            state["last_final_time"] = time.monotonic()
                            state["last_final_refined"] = False
                        state["committed"] = ""
                        state["last_words"] = []
                        state["stream_hyp_chars"] = 0

                def _handle(event: LiveDictationEvent) -> None:
                    try:
                        name = event.event
                        text = (event.text or "").strip()
                        if name in ("ready", "error", "stopped"):
                            print(f"[dictation] stream event: {name} text={text!r}")
                            if name == "ready":
                                try:
                                    self.dictation_stream_ready.emit(
                                        self._dictation_backend_label()
                                    )
                                except Exception:
                                    pass
                            return
                        if name == "hypothesis":
                            if not text:
                                return
                            # Drop deterministic whisper hallucinations BEFORE they
                            # can be live-typed. The final/rejected branch already
                            # strips these; the live path must too, or a low-signal
                            # partial types 'Thank you.'/'you' into the document.
                            # temperature=0 makes hallucinations repeat identically,
                            # so LocalAgreement-2 would otherwise happily commit them.
                            text = _strip_whisper_hallucinations(text)
                            if not text:
                                return
                            # A hypothesis arriving while a previous utterance's
                            # final is still buffered means a NEW utterance has
                            # begun (the streamer emits one final per utterance,
                            # then resets). Commit the previous one now — reconciling
                            # its live text — so this utterance's prefix bookkeeping
                            # (committed / stream_hyp_chars / last_words) starts clean.
                            # Done OUTSIDE the lock; _flush_pending re-takes it.
                            if state.get("pending_final_text"):
                                _flush_pending()
                            state["last_hypothesis_time"] = time.monotonic()
                            # The whole read-compute-type-write section is under
                            # _corrector_lock so a concurrent _commit_final (timer
                            # thread) can't interleave the shared char bookkeeping.
                            with self._corrector_lock:
                                words = text.split()
                                stable = _common_prefix(state["last_words"], words)
                                state["last_words"] = words
                                commit_words = stable[:-hold_back] if len(stable) > hold_back else []
                                if not commit_words:
                                    return
                                commit_text = " ".join(commit_words)
                                committed = state["committed"]
                                if commit_text.startswith(committed) and len(commit_text) > len(committed):
                                    suffix = commit_text[len(committed):]
                                    # On the FIRST insert of a new utterance, fix the
                                    # seam: re-case the leading word (whisper caps each
                                    # utterance's first word) and fill a dropped
                                    # boundary space. `cased` is length-preserving so
                                    # the char bookkeeping (stream_hyp_chars ==
                                    # len(committed)) is untouched; the lead space is a
                                    # separator and is deliberately NOT counted.
                                    lead, cased = "", suffix
                                    if committed == "":
                                        lead, cased = _seam_normalize(
                                            suffix, state.get("final_display", ""), state.get("last_char", "")
                                        )
                                    # SendInput (prefer_paste=False): live suffixes
                                    # are short and fire ~1.5x/sec — pasting each one
                                    # would clobber the user's clipboard continuously.
                                    inserted = self.text_input_controller.insert_text(lead + cased, prefer_paste=False)
                                    if inserted:
                                        self.grammar_corrector.append(lead + cased)
                                        state["committed"] = commit_text
                                        state["stream_hyp_chars"] = state.get("stream_hyp_chars", 0) + len(cased)
                                        state["last_char"] = cased[-1] if cased else state.get("last_char", "")
                            return
                        if name in ("final", "rejected"):
                            if not text:
                                _terminate_live_utterance()
                                return
                            text = _strip_whisper_hallucinations(text)
                            if not text:
                                _terminate_live_utterance()
                                return
                            command = _parse_dictation_command(text)
                            if command is not None:
                                _flush_pending()
                                newline_text = "\n\n" if command == "paragraph" else "\n"
                                with self._corrector_lock:
                                    inserted = self.text_input_controller.insert_text(
                                        newline_text, prefer_paste=False
                                    )
                                    if inserted:
                                        self.grammar_corrector.append(newline_text)
                                        state["last_char"] = "\n"
                                print(f"[dictation] command: {command} inserted={inserted} text={text!r}")
                                state["committed"] = ""
                                state["last_words"] = []
                                state["last_final_text"] = ""
                                state["stream_hyp_chars"] = 0
                                return
                            _buffer_final(text)
                    except Exception:
                        # Never let a handler exception kill the stream
                        pass

                streamer = self.live_dictation_streamer
                # System.Speech occasionally ends its Multiple-mode session on
                # its own (timeouts, audio device hiccups). Restart it until
                # the user actually toggles dictation off.
                try:
                    while True:
                        if stop_event is not None and stop_event.is_set():
                            break
                        # Flush any still-pending final BEFORE clearing the live
                        # prefix state, so a stream restart can't strand chars the
                        # hypothesis path already typed (which a later reconcile
                        # would then backspace against the wrong position).
                        if state.get("pending_final_text"):
                            _flush_pending()
                        # Reset per-utterance commit state so a fresh stream
                        # doesn't try to diff against stale words.
                        state["committed"] = ""
                        state["last_words"] = []
                        state["stream_hyp_chars"] = 0
                        ok = streamer.stream(stop_event=stop_event, event_callback=_handle)
                        if stop_event is not None and stop_event.is_set():
                            break
                        if not ok:
                            final_message = streamer.message or final_message
                            break
                        # Stream returned cleanly without stop_event -- loop and restart.
                except Exception as exc:
                    final_message = f"dictation error: {exc}"

                if stop_event is None or not stop_event.is_set():
                    self._voice_queue.put(
                        (
                            request_id,
                            {
                                "event": "dictation_complete",
                                "success": bool(state["final_display"]),
                                "heard_text": "",
                                "control_text": final_message,
                                "display_text": state["final_display"],
                            },
                        )
                    )

            self._voice_thread = threading.Thread(target=_dictation_worker, name="hgr-app-dictation", daemon=True)
            self._voice_thread.start()
            return

        if mode == "save_prompt":
            self.voice_status_overlay.show_listening(
                "Listening...",
                hint_text=self._save_prompt_text,
            )
            self.command_detected.emit(self._save_prompt_text)
            self._emit_status("save prompt active")
        elif mode == "selection" and self._selection_prompt_active:
            self.voice_status_overlay.update_selection_status("Listening...")
        elif mode == "create_playlist":
            self.voice_status_overlay.show_listening(
                "Listening...",
                hint_text="What would you like to call the playlist?",
            )
        elif mode == "add_playlist":
            self.voice_status_overlay.show_listening(
                "Listening...",
                hint_text="Which playlist should I add to?",
            )
        elif mode == "remove_playlist":
            self.voice_status_overlay.show_listening(
                "Listening...",
                hint_text="Which playlist should I remove from?",
            )
        else:
            self.voice_status_overlay.show_listening()
        if mode != "save_prompt":
            self._emit_status("voice listening...")

        chrome_mode_voice = self._chrome_mode_enabled and self.chrome_controller.is_window_open()
        preferred_target = "chrome" if chrome_mode_voice and mode == "general" else preferred_app

        def _push_status(status: str, *, command_text: str = "") -> None:
            self._voice_queue.put((request_id, {"event": "status", "status": status, "command_text": command_text}))

        def _worker() -> None:
            payload: dict | None = None
            heard_text = ""
            try:
                if mode == "save_prompt":
                    transcript_mode = "save_prompt"
                    listen_seconds = 9.5
                else:
                    transcript_mode = "playlist" if mode in {"add_playlist", "remove_playlist", "create_playlist"} else "command"
                    # Budget = start-wait (5s) + utterance headroom (~4s)
                    # + end-silence (3s) for commands. The old 5.0s was
                    # a total cap too tight for any of those â€” the loop
                    # ran out of blocks before the silence check could
                    # fire, so recordings cut after ~2s of voice audio.
                    listen_seconds = 8.5 if transcript_mode == "playlist" else 12.0
                result = self.voice_listener.listen(
                    max_seconds=listen_seconds,
                    status_callback=_push_status,
                    transcript_mode=transcript_mode,
                )
                heard_text = result.heard_text
                # Engine-log: surface what the listener heard, even if
                # downstream processing fails or finds no intent. Gives
                # the user immediate feedback that the mic IS picking
                # them up and helps debug "I said X but nothing
                # happened" reports.
                if heard_text and heard_text.strip():
                    try:
                        self.engine_log.emit(f"Voice heard: \"{heard_text.strip()[:120]}\"")
                    except Exception:
                        pass
                elif result.success is False:
                    try:
                        self.engine_log.emit("Voice listen ended (nothing heard)")
                    except Exception:
                        pass
                if mode in {"general", "selection"} and result.success:
                    _push_status("processing", command_text=result.heard_text)
                    context = VoiceCommandContext(preferred_app=preferred_target) if mode == "general" else None
                    try:
                        execution = self.voice_processor.execute(result.heard_text, context=context)
                        # Engine-log: whether the processor matched an
                        # intent. Distinguishes "we heard you but
                        # didn't understand" from "we heard you and
                        # did X" — both flow through this path today
                        # with no visible difference in the log.
                        try:
                            ex_intent = getattr(execution, "intent", None)
                            ex_target = str(getattr(execution, "target", "") or "")
                            ex_success = bool(getattr(execution, "success", False))
                            if ex_intent is None:
                                self.engine_log.emit(
                                    f"Voice command not recognized: \"{heard_text.strip()[:80]}\""
                                )
                            else:
                                intent_app = str(getattr(ex_intent, "app_name", "") or "")
                                intent_action = str(getattr(ex_intent, "action", "") or "")
                                verb = "executed" if ex_success else "attempted (failed)"
                                detail = (
                                    f"{intent_app} {intent_action}".strip()
                                    or ex_target
                                    or "command"
                                )
                                self.engine_log.emit(f"Voice command {verb}: {detail}")
                        except Exception:
                            pass
                        try:
                            from ...telemetry import track as _track
                            # Pull intent app + action so the dashboard
                            # can group "play X on YouTube" / "search X
                            # on Chrome" by app even though the
                            # engine-side `target` for both is just
                            # "chrome" (YouTube goes through the chrome
                            # router). intent_app_name distinguishes
                            # them: chrome / youtube / spotify / system /
                            # file_explorer / outlook / touchless.
                            intent = getattr(execution, "intent", None)
                            intent_app = ""
                            intent_action = ""
                            if intent is not None:
                                try:
                                    intent_app = str(getattr(intent, "app_name", "") or "")
                                    intent_action = str(getattr(intent, "action", "") or "")
                                except Exception:
                                    pass
                            _track(
                                "voice_command_executed",
                                {
                                    "target": str(getattr(execution, "target", "") or ""),
                                    "intent_app": intent_app,
                                    "intent_action": intent_action,
                                    "success": bool(getattr(execution, "success", False)),
                                    "in_tutorial": bool(self._tutorial_mode_enabled),
                                },
                            )
                        except Exception:
                            pass
                    except Exception as exc:
                        traceback.print_exc()
                        payload = {
                            "event": "result",
                            "mode": mode,
                            "success": False,
                            "target": "voice",
                            "heard_text": result.heard_text,
                            "control_text": f"voice command failed: {type(exc).__name__}",
                            "info_text": (str(exc) or "see console")[:200],
                            "display_text": result.heard_text,
                        }
                    else:
                        # Propagate the parsed intent's app_name + action
                        # so the result-handling code can show a more
                        # specific overlay (e.g. "Launching Spotify"
                        # instead of generic "Executing command") when
                        # the user just said something like "open spotify".
                        intent_app_name = ""
                        intent_action = ""
                        try:
                            if execution.intent is not None:
                                intent_app_name = str(execution.intent.app_name or "")
                                intent_action = str(execution.intent.action or "")
                        except Exception:
                            intent_app_name = ""
                            intent_action = ""
                        # Voice "clip that" — fire the same utility
                        # request the gesture path uses, but tag it
                        # with the _voice suffix so the UI knows to
                        # skip the save-location prompt and the
                        # multi-monitor picker (the user just said
                        # "clip that"; nagging them with a follow-up
                        # would defeat the point).
                        if (
                            execution.success
                            and intent_app_name == "touchless"
                            and intent_action in {
                                "clip_1m", "clip_30s", "clip_2m", "clip_5m",
                                "clip_default",
                            }
                        ):
                            # Resolve clip_default to a concrete length
                            # using the user's preset preference. Keeps
                            # the voice parser config-agnostic.
                            if intent_action == "clip_default":
                                try:
                                    preset_seconds = int(
                                        getattr(self.config, "clip_default_duration_seconds", 60)
                                        or 60
                                    )
                                except Exception:
                                    preset_seconds = 60
                                if preset_seconds <= 30:
                                    intent_action = "clip_30s"
                                elif preset_seconds <= 90:
                                    intent_action = "clip_1m"
                                elif preset_seconds <= 180:
                                    intent_action = "clip_2m"
                                else:
                                    intent_action = "clip_5m"
                            # Anchor the clip window at when the user
                            # actually finished saying "clip that", not
                            # at the much-later moment when this branch
                            # fires. Whisper + parse + dispatch easily
                            # add 5-15 s; without this the saved clip
                            # ends 5-15 s after the user's intent and
                            # the moment they wanted to capture has
                            # already scrolled out the back of the
                            # rolling buffer. main_window reads this
                            # attribute when processing the matching
                            # _voice utility request.
                            try:
                                self._pending_clip_voice_end_ts = (
                                    float(result.speech_end_ts)
                                    if getattr(result, "speech_end_ts", None) is not None
                                    else None
                                )
                            except (TypeError, ValueError):
                                self._pending_clip_voice_end_ts = None
                            try:
                                import sys as _sys, time as _time
                                _sys.stderr.write(
                                    f"[clip-anchor] engine stashed end_ts={self._pending_clip_voice_end_ts} "
                                    f"(now={_time.time():.3f}, "
                                    f"lag_since_speech_end="
                                    f"{(_time.time() - self._pending_clip_voice_end_ts):.2f}s)"
                                    if self._pending_clip_voice_end_ts is not None
                                    else
                                    f"[clip-anchor] engine stashed end_ts=None (speech_end_ts missing from listener result)"
                                )
                                _sys.stderr.write("\n")
                                _sys.stderr.flush()
                            except Exception:
                                pass
                            try:
                                self._queue_utility_request(f"{intent_action}_voice")
                            except Exception:
                                pass
                        payload = {
                            "event": "result",
                            "mode": mode,
                            "success": execution.success,
                            "target": execution.target,
                            "heard_text": execution.heard_text,
                            "control_text": execution.control_text,
                            "info_text": execution.info_text,
                            "display_text": getattr(execution, "display_text", execution.heard_text),
                            "intent_app_name": intent_app_name,
                            "intent_action": intent_action,
                        }
                else:
                    payload = {
                        "event": "result",
                        "mode": mode,
                        "success": result.success,
                        "target": "save_prompt" if mode == "save_prompt" else "voice",
                        "heard_text": result.heard_text,
                        "control_text": result.message,
                        "info_text": self._spotify_info_text,
                        "display_text": result.heard_text,
                    }
            except Exception as exc:
                traceback.print_exc()
                payload = {
                    "event": "result",
                    "mode": mode,
                    "success": False,
                    "target": "voice",
                    "heard_text": heard_text,
                    "control_text": f"voice worker crashed: {type(exc).__name__}",
                    "info_text": (str(exc) or "see console")[:200],
                    "display_text": heard_text or "voice error",
                }
            finally:
                if payload is None:
                    payload = {
                        "event": "result",
                        "mode": mode,
                        "success": False,
                        "target": "voice",
                        "heard_text": heard_text,
                        "control_text": "voice worker exited without result",
                        "info_text": "-",
                        "display_text": heard_text or "voice error",
                    }
                self._voice_queue.put((request_id, payload))

        self._voice_thread = threading.Thread(target=_worker, name="hgr-app-voice-command", daemon=True)
        self._voice_thread.start()

    def _drain_voice_results(self) -> None:
        while True:
            try:
                request_id, payload = self._voice_queue.get_nowait()
            except queue.Empty:
                break
            if request_id != self._voice_request_id:
                continue
            event = str(payload.get("event", "result"))
            if event == "status":
                status = str(payload.get("status", "") or "").strip().lower()
                command_text = str(payload.get("command_text", "") or "").strip()
                if command_text:
                    self._voice_heard_text = command_text
                    self._voice_display_text = command_text
                if status == "listening":
                    if self._voice_mode == "dictation":
                        self._voice_control_text = "dictation listening..."
                    elif self._voice_mode == "save_prompt":
                        self._voice_control_text = self._save_prompt_text
                    else:
                        self._voice_control_text = "voice listening..."
                    if self._voice_mode == "dictation":
                        self.voice_status_overlay.show_processing("Dictation active", command_text="")
                    elif self._voice_mode == "save_prompt":
                        self.voice_status_overlay.show_listening("Listening...", hint_text=self._save_prompt_text)
                    else:
                        if self._selection_prompt_active and getattr(self.voice_status_overlay, "_mode", "") == "selection":
                            self.voice_status_overlay.update_selection_status("Listening...")
                        else:
                            self.voice_status_overlay.show_listening()
                elif status == "recognizing":
                    if self._voice_mode == "dictation":
                        self._voice_control_text = "dictation active"
                        self.voice_status_overlay.show_processing("Dictation active", command_text="")
                    elif self._voice_mode == "save_prompt":
                        self._voice_control_text = "recognizing save location..."
                        self.voice_status_overlay.show_processing(
                            "Processing command...",
                            command_text=self._voice_display_text if self._voice_display_text != "-" else "",
                        )
                    else:
                        self._voice_control_text = "recognizing..."
                        if self._selection_prompt_active and getattr(self.voice_status_overlay, "_mode", "") == "selection":
                            self.voice_status_overlay.update_selection_status("Recognizing...")
                        else:
                            self.voice_status_overlay.show_processing(
                                "Recognizing...",
                                command_text=self._voice_display_text if self._voice_display_text != "-" else "",
                            )
                elif status == "processing":
                    if self._voice_mode == "save_prompt":
                        self._voice_control_text = "processing save location..."
                    else:
                        self._voice_control_text = "processing command..."
                    if self._selection_prompt_active and getattr(self.voice_status_overlay, "_mode", "") == "selection":
                        self.voice_status_overlay.hide_overlay()
                    elif self._voice_mode == "save_prompt":
                        self.voice_status_overlay.show_processing(
                            "Executing command...",
                            command_text=command_text or (self._voice_display_text if self._voice_display_text != "-" else ""),
                        )
                    else:
                        self.voice_status_overlay.show_processing(
                            "Processing command...",
                            command_text=command_text or (self._voice_display_text if self._voice_display_text != "-" else ""),
                        )
                continue
            if event == "dictation_chunk":
                heard_text = str(payload.get("heard_text", "") or "").strip()
                display_text = str(payload.get("display_text", "") or "").strip()
                partial = bool(payload.get("partial"))
                if heard_text:
                    self._voice_heard_text = heard_text
                if display_text:
                    self._voice_display_text = display_text
                self._voice_control_text = str(payload.get("control_text", "dictation updated"))
                if partial:
                    self._voice_control_text = "live dictating..."
                self.command_detected.emit(self._voice_control_text)
                self.voice_status_overlay.show_listening(backend_label=self._dictation_backend_label())
                continue
            if event == "dictation_complete":
                self._voice_listening = False
                self._dictation_active = False
                self._dictation_backend = "idle"
                self._voice_mode = "ready"
                self._voice_stop_event = None
                try:
                    self.grammar_corrector.stop()
                except Exception:
                    pass
                try:
                    if self.whisper_refiner is not None:
                        self.whisper_refiner.stop()
                except Exception:
                    pass
                self._dictation_state = None
                display_text = str(payload.get("display_text", "") or "").strip()
                if display_text:
                    self._voice_display_text = display_text
                self._voice_control_text = str(payload.get("control_text", "dictation stopped"))
                self.command_detected.emit(self._voice_control_text)
                self.voice_status_overlay.show_result(
                    "Dictation stopped",
                    command_text="",
                    duration=1.8,
                )
                continue
            self._voice_listening = False
            self._dictation_active = False
            self._dictation_backend = "idle"
            self._voice_mode = "ready"
            self._voice_stop_event = None
            try:
                self.grammar_corrector.stop()
            except Exception:
                pass
            try:
                if self.whisper_refiner is not None:
                    self.whisper_refiner.stop()
            except Exception:
                pass
            self._dictation_state = None
            mode = str(payload.get("mode", "general"))
            heard_text = str(payload.get("heard_text", "") or "").strip()
            self._voice_heard_text = heard_text or "-"
            self._voice_display_text = str(payload.get("display_text", "") or heard_text or "-")
            self._voice_control_text = str(payload.get("control_text", "voice idle"))
            target = payload.get("target")
            if mode == "general":
                if target == "spotify":
                    if payload.get("success"):
                        self._spotify_info_text = str(payload.get("info_text", self._spotify_info_text))
                    self._spotify_control_text = str(payload.get("control_text", self._spotify_control_text))
                elif target == "chrome":
                    self._chrome_control_text = str(payload.get("control_text", self._chrome_control_text))
                if target == "voice_selection":
                    self.command_detected.emit(self._voice_control_text)
                    title, items, instruction = self._parse_voice_selection_text(self._voice_display_text if self._voice_display_text != "-" else "")
                    self._selection_prompt_active = True
                    self._selection_prompt_title = title
                    self._selection_prompt_items = items
                    self._selection_prompt_instruction = instruction
                    self.voice_status_overlay.show_selection(title, items, instruction, status_text="")
                    continue
                self.command_detected.emit(self._voice_control_text)
                # Customize the overlay headline based on the intent
                # the voice processor parsed. For 'open'/'launch'-style
                # commands targeted at a known app, say "Launching X"
                # instead of the generic "Executing command".
                if payload.get("success"):
                    headline = self._build_launch_overlay_headline(payload)
                else:
                    # Failed but had a parsed intent: surface the
                    # controller's actual message (e.g. "spotify
                    # window not found", "spotify launch path not
                    # found") instead of the misleading generic
                    # "Command not understood" â€” which conflated a
                    # transcription failure with an execution
                    # failure on a perfectly-recognized command.
                    intent_app = str(payload.get("intent_app_name") or "").strip()
                    control_text = str(payload.get("control_text") or "").strip()
                    if intent_app and control_text:
                        headline = control_text
                    else:
                        headline = "Command not understood"
                self.voice_status_overlay.show_result(
                    headline,
                    command_text=self._voice_display_text if self._voice_display_text != "-" else "",
                    duration=2.0,
                )
                continue

            if mode == "selection":
                if target == "voice_selection":
                    self.command_detected.emit(self._voice_control_text)
                    title, items, instruction = self._parse_voice_selection_text(self._voice_display_text if self._voice_display_text != "-" else "")
                    self._selection_prompt_active = True
                    self._selection_prompt_title = title
                    self._selection_prompt_items = items
                    self._selection_prompt_instruction = instruction
                    self.voice_status_overlay.show_selection(title, items, instruction, status_text="")
                    continue
                self._selection_prompt_active = False
                if str(self._voice_control_text).lower().startswith("selection cancelled"):
                    self.command_detected.emit(self._voice_control_text)
                    self.voice_status_overlay.show_result("Selection cancelled", command_text="", duration=1.4)
                    continue
                self.command_detected.emit(self._voice_control_text)
                self.voice_status_overlay.show_result(
                    "Executing command" if payload.get("success") else "Command not understood",
                    command_text=self._voice_display_text if self._voice_display_text != "-" else "",
                    duration=2.0,
                )
                continue

            if mode == "save_prompt":
                self._save_prompt_active = False
                self.command_detected.emit("save destination received" if heard_text else "using default save location")
                self.voice_status_overlay.show_result(
                    "Save preference received" if heard_text else "Using default save folder",
                    command_text=self._voice_display_text if self._voice_display_text != "-" else "",
                    duration=1.5,
                )
                self.save_prompt_completed.emit(
                    {
                        "success": bool(payload.get("success")),
                        "heard_text": heard_text,
                        "control_text": self._voice_control_text,
                        "display_text": self._voice_display_text,
                    }
                )
                continue

            if not payload.get("success"):
                self._spotify_control_text = str(payload.get("control_text", "voice command not understood"))
                self.command_detected.emit(self._spotify_control_text)
                self.voice_status_overlay.show_result(
                    "Playlist not understood",
                    command_text=self._voice_display_text if self._voice_display_text != "-" else "",
                    duration=2.0,
                )
                continue

            if mode == "create_playlist":
                playlist_name = self._clean_create_playlist_reply(heard_text)
            else:
                playlist_name = self._clean_playlist_reply(heard_text)
            if not playlist_name:
                self._spotify_control_text = "playlist name not understood"
                self.command_detected.emit(self._spotify_control_text)
                self.voice_status_overlay.show_result(
                    "Playlist not understood",
                    command_text=self._voice_display_text if self._voice_display_text != "-" else "",
                    duration=2.0,
                )
                continue

            if mode == "add_playlist":
                success = self.spotify_controller.add_current_track_to_playlist(playlist_name)
            elif mode == "create_playlist":
                success = self.spotify_controller.create_playlist(playlist_name)
            else:
                success = self.spotify_controller.remove_current_track_from_playlist(playlist_name)
            self._spotify_control_text = self.spotify_controller.message
            self.command_detected.emit(self._spotify_control_text)
            if success:
                details = self.spotify_controller.get_current_track_details()
                if details is not None:
                    self._spotify_info_text = details.summary()
            self.voice_status_overlay.show_result(
                self.spotify_controller.message,
                command_text=playlist_name,
                duration=2.0,
            )

    def _parse_voice_selection_text(self, text: str) -> tuple[str, list[tuple[str, str, str]], str]:
        lines = [line.strip() for line in str(text or "").splitlines() if line.strip()]
        title = "Choose a file or folder"
        if lines and "app" in lines[0].lower():
            title = "Choose a file or app"
        instruction = "Say the corresponding letter"
        items: list[tuple[str, str, str]] = []
        for line in lines:
            # Accept A-Z keys AND the new A1/B1/Z5 cycle-suffix keys
            # produced past index 25 by _selection_key_for_index.
            # Prior `[A-Za-z]` (single char) rejected anything past Z.
            match = re.match(r"^([A-Za-z]\d*|\d+)\.\s+(.*?)(?:\s+[—-]\s+(.*))?$", line)
            if match:
                selection_key = str(match.group(1) or "").strip().upper()
                label = str(match.group(2) or "").strip()
                path_text = str(match.group(3) or "").strip()
                items.append((selection_key, label, path_text))
                continue
            if line.lower().startswith("say "):
                instruction = line
            elif line.lower().startswith("which "):
                instruction = line
        return title, items, instruction

    def _on_dictation_stream_ready(self, backend_label: str) -> None:
        try:
            self.voice_status_overlay.show_listening(backend_label=backend_label)
        except Exception:
            pass

    def _handle_voice_overlay_selection(self, selection_key: str) -> None:
        try:
            result = self.voice_processor.execute(str(selection_key or "").strip())
        except Exception:
            return
        self._voice_queue.put((self._voice_request_id, {
            "event": "result",
            "mode": "general",
            "success": result.success,
            "target": result.target,
            "heard_text": result.heard_text,
            "control_text": result.control_text,
            "info_text": result.info_text,
            "display_text": getattr(result, "display_text", result.heard_text),
        }))

    def _update_spotify_wheel_overlay(self, now: float) -> None:
        if not self._spotify_wheel_visible:
            if self.spotify_wheel_overlay.isVisible():
                self.spotify_wheel_overlay.hide_overlay()
            return
        selected_label = "Hold a slice"
        if self._spotify_wheel_selected_key is not None:
            selected_label = self._spotify_wheel_label(self._spotify_wheel_selected_key)
        selection_progress = 0.0
        if self._spotify_wheel_selected_key is not None:
            selection_progress = min(1.0, max(0.0, (now - self._spotify_wheel_selected_since) / 1.0))
        self.spotify_wheel_overlay.set_wheel(
            items=self._spotify_wheel_items(),
            selected_key=self._spotify_wheel_selected_key,
            selection_progress=selection_progress,
            status_text=selected_label,
            cursor_offset=self._spotify_wheel_cursor_offset,
        )
        self.spotify_wheel_overlay.show_overlay()

    def _update_chrome_wheel_overlay(self, now: float) -> None:
        if not self._chrome_wheel_visible:
            if self.chrome_wheel_overlay.isVisible():
                self.chrome_wheel_overlay.hide_overlay()
            return
        selected_label = "Hold a slice"
        if self._chrome_wheel_selected_key is not None:
            selected_label = self._chrome_wheel_label(self._chrome_wheel_selected_key)
        selection_progress = 0.0
        if self._chrome_wheel_selected_key is not None:
            selection_progress = min(1.0, max(0.0, (now - self._chrome_wheel_selected_since) / 1.0))
        self.chrome_wheel_overlay.set_wheel(
            items=self._chrome_wheel_items(),
            selected_key=self._chrome_wheel_selected_key,
            selection_progress=selection_progress,
            status_text=selected_label,
            cursor_offset=self._chrome_wheel_cursor_offset,
        )
        self.chrome_wheel_overlay.show_overlay()

    def _update_youtube_wheel_overlay(self, now: float) -> None:
        if not self._youtube_wheel_visible:
            if self.youtube_wheel_overlay.isVisible():
                self.youtube_wheel_overlay.hide_overlay()
            return
        selected_label = "Hold a slice"
        if self._youtube_wheel_selected_key is not None:
            selected_label = self._youtube_wheel_label(self._youtube_wheel_selected_key)
        selection_progress = 0.0
        if self._youtube_wheel_selected_key is not None:
            selection_progress = min(1.0, max(0.0, (now - self._youtube_wheel_selected_since) / 1.0))
        self.youtube_wheel_overlay.set_wheel(
            items=self._youtube_wheel_items(),
            selected_key=self._youtube_wheel_selected_key,
            selection_progress=selection_progress,
            status_text=selected_label,
            cursor_offset=self._youtube_wheel_cursor_offset,
        )
        self.youtube_wheel_overlay.show_overlay()

    def _update_drawing_wheel_overlay(self, now: float) -> None:
        if not self._drawing_wheel_visible:
            if self.drawing_wheel_overlay.isVisible():
                self.drawing_wheel_overlay.hide_overlay()
            return
        selected_label = "Hold a slice"
        if self._drawing_wheel_selected_key is not None:
            selected_label = self._drawing_wheel_label(self._drawing_wheel_selected_key)
        selection_progress = 0.0
        if self._drawing_wheel_selected_key is not None:
            selection_progress = min(1.0, max(0.0, (now - self._drawing_wheel_selected_since) / 0.85))
        self.drawing_wheel_overlay.set_wheel(
            items=self._drawing_wheel_items(),
            selected_key=self._drawing_wheel_selected_key,
            selection_progress=selection_progress,
            status_text=selected_label,
            cursor_offset=self._drawing_wheel_cursor_offset,
        )
        self.drawing_wheel_overlay.show_overlay()

    def _update_utility_wheel_overlay(self, now: float) -> None:
        if not self._utility_wheel_visible:
            if self.utility_wheel_overlay.isVisible():
                self.utility_wheel_overlay.hide_overlay()
            return
        selected_label = "Hold a slice"
        if self._utility_wheel_selected_key is not None:
            selected_label = self._utility_wheel_label(self._utility_wheel_selected_key)
        selection_progress = 0.0
        if self._utility_wheel_selected_key is not None:
            selection_progress = min(1.0, max(0.0, (now - self._utility_wheel_selected_since) / 0.85))
        self.utility_wheel_overlay.set_wheel(
            items=self._utility_wheel_items(),
            selected_key=self._utility_wheel_selected_key,
            selection_progress=selection_progress,
            status_text=selected_label,
            cursor_offset=self._utility_wheel_cursor_offset,
        )
        self.utility_wheel_overlay.show_overlay()

    def _cancel_all_voice_stages(self) -> None:
        was_active = bool(
            self._voice_listening
            or self._dictation_active
            or self._save_prompt_active
            or self._selection_prompt_active
        )
        was_save_prompt = bool(self._save_prompt_active)
        self._reset_voice_state()
        if was_save_prompt:
            try:
                self.save_prompt_completed.emit(
                    {
                        "success": False,
                        "heard_text": "",
                        "control_text": "save prompt canceled",
                        "display_text": "-",
                        "canceled": True,
                    }
                )
            except Exception:
                pass
        if was_active:
            self._voice_control_text = "voice canceled"
            try:
                self.command_detected.emit(self._voice_control_text)
            except Exception:
                pass
            try:
                self.voice_status_overlay.show_result("Canceled", command_text="", duration=1.0)
            except Exception:
                pass
            self._record_action("voice_cancel", "voice canceled")

    def _reset_voice_state(self) -> None:
        if self._voice_stop_event is not None:
            self._voice_stop_event.set()
            self._voice_stop_event = None
        self.dictation_processor.reset()
        try:
            self.grammar_corrector.stop()
        except Exception:
            pass
        try:
            if self.whisper_refiner is not None:
                self.whisper_refiner.stop()
        except Exception:
            pass
        self._dictation_state = None
        self._voice_candidate = "neutral"
        self._voice_candidate_since = 0.0
        self._voice_cooldown_until = 0.0
        self._voice_latched_label = None
        self._voice_request_id += 1
        self._voice_listening = False
        self._dictation_active = False
        self._dictation_backend = "idle"
        self._voice_mode = "ready"
        self._voice_control_text = self.voice_listener.message
        self._voice_heard_text = "-"
        self._voice_display_text = "-"
        self._dictation_toggle_release_required = False
        self._dictation_stop_rearm_at = 0.0
        self._dictation_release_candidate_since = 0.0
        self._selection_prompt_active = False
        self._selection_prompt_title = "Which file/folder?"
        self._selection_prompt_items = []
        self._selection_prompt_instruction = "Say the corresponding letter."
        self._save_prompt_active = False
        self.voice_status_overlay.hide_overlay()
        while True:
            try:
                self._voice_queue.get_nowait()
            except queue.Empty:
                break

    def start_save_location_prompt(self) -> bool:
        if not self._running:
            return False
        if self._voice_listening:
            self._reset_voice_state()
        self._save_prompt_active = True
        self._start_voice_capture(mode="save_prompt", preferred_app=None)
        return True

# Author: Konstantin Markov

