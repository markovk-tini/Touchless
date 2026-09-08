from __future__ import annotations

import json
import os
import platform
import re
import shutil
import subprocess
import tempfile
import time
import wave
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Optional

import numpy as np

from ..utils.runtime_paths import app_base_path
from ..utils.subprocess_utils import hidden_subprocess_kwargs


@dataclass(frozen=True)
class VoiceCommandResult:
    heard_text: str
    success: bool
    message: str
    # Wall-clock time.time() at the moment VAD declared end-of-speech
    # (i.e. when the user actually finished saying the command). This
    # is captured BEFORE whisper inference, parse, and dispatch — so
    # downstream consumers ("clip that" especially) can anchor their
    # work to "when the user spoke" instead of "when execute() got
    # around to firing", which differs by 4-10+ seconds in practice
    # and shifts time-window-based actions (clip-the-last-60s) out of
    # alignment with what the user wanted to capture.
    speech_end_ts: float | None = None


def _live_windows_input_endpoints_via_pycaw() -> list[str]:
    """Enumerate ACTIVE Windows audio-capture endpoints via pycaw
    (Windows Core Audio). This bypasses PortAudio's PA_Initialize
    device cache — sounddevice/PortAudio snapshots the device list
    once at startup and never re-scans, so mid-session hotplug
    (headset plugged in after Touchless launched) is invisible to
    sd.query_devices() until process restart. pycaw's GetAllDevices
    calls IMMDeviceEnumerator::EnumAudioEndpoints directly which
    always returns live data.

    v1.1.7 fix (round 8): the previous implementation guessed at a
    ``.direction`` attribute that pycaw's AudioDevice does not
    expose, and called ``int(dev.state)`` on a plain Enum which
    raises TypeError. Both errors were swallowed by a broad
    try/except so every device was silently dropped — the function
    returned [] deterministically and dad's hot-plug bug was never
    touched. Correct call: pass eCapture + ACTIVE at the COM layer
    so no Python-side filtering is needed. Verified live to return
    exactly the real capture endpoints Windows Sound shows.

    Returns [] on any failure so callers fall back to the
    sounddevice enumeration path.
    """
    try:
        from pycaw.pycaw import AudioUtilities  # type: ignore
        from pycaw.constants import EDataFlow, DEVICE_STATE  # type: ignore
    except Exception:
        return []
    # Suppress pycaw's occasional COMError-during-property-fetch
    # UserWarnings — the diff timer would spam stderr every 3 s
    # otherwise. Real errors still propagate.
    import warnings as _warnings
    try:
        with _warnings.catch_warnings():
            _warnings.simplefilter("ignore")
            devs = AudioUtilities.GetAllDevices(
                data_flow=EDataFlow.eCapture.value,
                device_state=DEVICE_STATE.ACTIVE.value,
            )
    except Exception:
        return []
    names: list[str] = []
    seen: set[str] = set()
    for dev in devs or []:
        try:
            name = str(getattr(dev, "FriendlyName", "") or "").strip()
        except Exception:
            continue
        if not name or name in seen:
            continue
        seen.add(name)
        names.append(name)
    return names


def list_input_microphones() -> list[str]:
    """Return readable names for available input-capable microphone devices.

    On Windows, sounddevice/PortAudio enumerates every device under
    EACH host API (MME / DirectSound / WASAPI / WDM-KS), so a single
    physical mic shows up 4× — sometimes with slightly different
    name suffixes that the dedup-by-name step missed. The user
    reported "I have 3 real mics in Windows settings but the
    Touchless dropdown shows 8+ entries". Fix: restrict the listing
    to WASAPI on Windows because that's the host API the Windows
    Sound control panel uses, so the dropdown matches what the user
    sees there. Other platforms fall back to the previous all-API
    listing (where the duplicate-host-API problem doesn't exist).

    v1.1.7 hot-plug fix: sounddevice caches the device list at
    PA_Initialize and does NOT re-scan on subsequent query_devices()
    calls, so a headset plugged in mid-session was invisible until
    process restart. Now we FIRST try pycaw (Windows Core Audio)
    which always returns live data, then union with the sounddevice
    result. Union guarantees no regression when pycaw is missing
    or returns a subset.
    """
    live_windows_names = _live_windows_input_endpoints_via_pycaw()

    try:
        import sounddevice as sd
    except Exception:
        return live_windows_names

    try:
        devices = sd.query_devices()
    except Exception:
        return live_windows_names

    # Identify the WASAPI host-api index on Windows. On other
    # platforms we leave wasapi_index=None and the filter no-ops.
    wasapi_index: int | None = None
    try:
        if platform.system() == "Windows":
            for idx, host in enumerate(sd.query_hostapis()):
                host_name = str(host.get("name", "") or "").strip().lower()
                if host_name == "windows wasapi":
                    wasapi_index = idx
                    break
    except Exception:
        wasapi_index = None

    names: list[str] = []
    seen: set[str] = set()
    for device in devices:
        try:
            max_inputs = int(device.get("max_input_channels", 0) or 0)
        except Exception:
            max_inputs = 0
        if max_inputs <= 0:
            continue
        # On Windows, only accept devices exposed through WASAPI.
        # That's the host API the Windows Sound control panel +
        # modern apps use; restricting to it eliminates the 3-4×
        # duplication caused by the legacy MME / DirectSound /
        # WDM-KS exposures of the same physical hardware.
        if wasapi_index is not None:
            try:
                if int(device.get("hostapi", -1)) != wasapi_index:
                    continue
            except Exception:
                continue
        name = str(device.get("name", "") or "").strip()
        if not name or name in seen:
            continue
        seen.add(name)
        names.append(name)
    # Defensive fallback: if the WASAPI filter eliminated every
    # device (e.g. PortAudio compiled without WASAPI support on
    # this user's setup), fall back to the legacy unfiltered list
    # so the dropdown isn't empty — better to show duplicates than
    # to lock the user out of mic selection entirely.
    if not names and wasapi_index is not None:
        for device in devices:
            try:
                max_inputs = int(device.get("max_input_channels", 0) or 0)
            except Exception:
                max_inputs = 0
            if max_inputs <= 0:
                continue
            name = str(device.get("name", "") or "").strip()
            if not name or name in seen:
                continue
            seen.add(name)
            names.append(name)
    # v1.1.7 hot-plug union: merge any pycaw-discovered live
    # Windows endpoints that sounddevice didn't see (they might
    # be brand-new hot-plug devices that PortAudio's PA_Initialize
    # cache doesn't know about yet). This is the union that fixes
    # dad's headset-not-appearing bug.
    for live_name in live_windows_names:
        try:
            live_name = str(live_name or "").strip()
        except Exception:
            continue
        if not live_name or live_name in seen:
            continue
        seen.add(live_name)
        names.append(live_name)
    return names


def _dedup_repeated_phrase(text: str) -> str:
    """Collapse near-duplicate phrases whisper sometimes emits when
    trailing silence triggers a hallucinated repeat. Examples handled:

        "play poker face, play poker face"           -> "play poker face"
        "play poekr face, play poker face"           -> "play poker face"   (picks longer of two near-duplicates)
        "set volume to 30. set volume to 30"         -> "set volume to 30"
        "open chrome and open chrome"                -> "open chrome and open chrome"  (left alone — has connector)

    Heuristic: split on commas / sentence punctuation, compute pairwise
    similarity, drop the shorter (likely truncated) of any pair > 0.75
    similar. Conservative — does nothing when fragments are clearly
    different commands."""
    if not text:
        return text
    import re as _re
    # Split on commas, periods, semicolons followed by space.
    parts = [p.strip() for p in _re.split(r"\s*[,;]\s*|\s*\.\s+", text) if p.strip()]
    if len(parts) < 2:
        return text
    # Compare adjacent fragments. Two similarity signals:
    #   (a) token-set overlap — catches reordering / extra words.
    #   (b) character-level SequenceMatcher — catches single-character
    #       typos that survive token-set ('poekr' vs 'poker' = 2 of 3
    #       tokens match = 0.67, but char ratio is 0.93).
    # Either >= 0.75 triggers dedup. Keep the longer fragment (more
    #   likely to be the complete + correct transcription).
    import difflib as _difflib

    def _similar(a: str, b: str) -> float:
        a_low, b_low = a.lower(), b.lower()
        ta = set(_re.findall(r"[a-z0-9]+", a_low))
        tb = set(_re.findall(r"[a-z0-9]+", b_low))
        token_sim = (len(ta & tb) / max(len(ta), len(tb))
                     if ta and tb else 0.0)
        char_sim = _difflib.SequenceMatcher(None, a_low, b_low).ratio()
        return max(token_sim, char_sim)

    def _better(a: str, b: str) -> str:
        # 1. Length tiebreaker — much longer one is more complete.
        if abs(len(a) - len(b)) > 2:
            return a if len(a) > len(b) else b
        # 2. Reject suspicious consonant clusters (4+ consonants in a
        #    row almost never appear in real English words).
        bad_clusters = _re.compile(r"[bcdfghjklmnpqrstvwxz]{4,}", _re.IGNORECASE)
        a_bad = bool(bad_clusters.search(a))
        b_bad = bool(bad_clusters.search(b))
        if a_bad and not b_bad:
            return b
        if b_bad and not a_bad:
            return a
        # 3. Whisper hallucinated-duplicate heuristic: when lengths are
        #    very close, the SECOND fragment is usually the cleaner
        #    one. The first pass got the typo ("poekr"), the second
        #    pass had more context and produced "poker". So pick `b`
        #    when a tie remains. Verified empirically on this user's
        #    reported failure ("play poekr face, play poker face").
        return b

    kept: list = [parts[0]]
    for p in parts[1:]:
        if _similar(kept[-1], p) >= 0.75:
            kept[-1] = _better(kept[-1], p)
        else:
            kept.append(p)
    return ", ".join(kept)


def _audio_file_duration_sec(path: Path) -> float:
    """Quick header-only WAV duration probe. Returns 0.0 on any error
    (caller falls back to longer-utterance decode params). Avoids
    librosa / scipy."""
    try:
        with wave.open(str(path), "rb") as wf:
            frames = wf.getnframes()
            rate = wf.getframerate() or 1
            return frames / float(rate)
    except Exception:
        return 0.0


def _preprocess_audio_for_whisper(
    audio: "np.ndarray", sample_rate: int,
    *,
    user_gain: float = 1.0,
) -> "np.ndarray":
    """Minimal preprocessing — just DC offset removal.

    History note: this function previously had bandpass + AGC +
    noise gate stages. ALL of them audibly garbled the WAV (the
    user's mic test in Touchless played back clean raw audio,
    proving the capture itself is fine — the corruption was
    happening here). The noise gate's hard-zero mask in particular
    produced step discontinuities that sound like garbled audio
    on playback AND register in Whisper's mel filterbank as
    spurious consonant transients. The AGC's RMS-targeted gain
    swings produced pumping artifacts. The bandpass at 80-7900
    added phase distortion that compounded with the resample.

    Whisper was trained on 680k hours of real-world unprocessed
    audio (web video, podcasts, calls). It handles raw input far
    better than aggressively-processed input — every stage we add
    on top of the clean capture is a chance to introduce
    artifacts the model wasn't trained for. Just DC removal +
    downstream peak normalize is the right amount of processing.

    Total cost is <0.1 ms for a 3 s buffer at 48 kHz."""
    if audio.size == 0:
        return audio
    mean = float(audio.mean())
    if abs(mean) > 1e-4:
        audio = audio - mean
    return audio.astype(np.float32, copy=False)


def _resample_to_16k(audio: "np.ndarray", src_rate: int) -> "np.ndarray":
    """Whisper's native input rate is 16 kHz; it down-samples internally
    if we hand it 48 kHz. Doing the resample ourselves is faster (~5x
    smaller WAV file → less disk I/O, less decode work inside whisper)
    AND removes whisper's resampler from the variability budget — we
    use scipy when present, polyphase numpy fallback when not.
    Returns audio at exactly 16000 Hz."""
    target_rate = 16000
    if src_rate == target_rate or audio.size == 0:
        return audio
    # Prefer scipy.signal.resample_poly — high-quality polyphase
    # filter, much better than linear interpolation.
    try:
        from scipy.signal import resample_poly  # type: ignore
        from math import gcd
        g = gcd(src_rate, target_rate)
        up, down = target_rate // g, src_rate // g
        return resample_poly(audio, up, down).astype(np.float32, copy=False)
    except Exception:
        pass
    # Fallback: linear interpolation. Lower quality but never fails.
    ratio = target_rate / src_rate
    n_out = max(1, int(round(audio.shape[0] * ratio)))
    x_old = np.linspace(0, 1, num=audio.shape[0], endpoint=False)
    x_new = np.linspace(0, 1, num=n_out, endpoint=False)
    return np.interp(x_new, x_old, audio).astype(np.float32, copy=False)


def _pick_whisper_device() -> tuple[str, str]:
    """Choose the fastest faster-whisper (device, compute_type) for this
    machine. Returns ("cuda", "int8_float16") on CUDA-capable systems,
    otherwise ("cpu", "int8"). Detection mirrors whisper_stream.py:
    look for an importable nvidia driver path via CTranslate2's CUDA
    presence, OR fall back to nvidia-smi probe. Never raises — failure
    just drops to CPU. The caller still retries with CPU if model load
    fails, so a wrong CUDA report can't lock the user out."""
    # 1. Cheapest check first: env override for users who want to force.
    forced = (os.environ.get("HGR_WHISPER_DEVICE") or "").strip().lower()
    if forced in ("cpu", "cuda"):
        return (forced,
                "int8_float16" if forced == "cuda" else "int8")
    # 2. CTranslate2 (faster-whisper's runtime) ships per-device wheels;
    # the GPU wheel exposes get_supported_compute_types("cuda") cleanly.
    try:
        import ctranslate2  # type: ignore
        supported = ctranslate2.get_supported_compute_types("cuda")
        if supported:
            # int8_float16 is the sweet spot: ~2-3x speedup vs CPU int8
            # with negligible quality loss. float16 alone needs ~3x VRAM.
            ctype = "int8_float16" if "int8_float16" in supported \
                else ("float16" if "float16" in supported
                      else "default")
            return "cuda", ctype
    except Exception:
        pass
    # 3. Fallback probe: nvidia-smi. Slower (subprocess) but tolerant
    # of broken CTranslate2 installs.
    try:
        if shutil.which("nvidia-smi"):
            r = subprocess.run(
                ["nvidia-smi", "-L"],
                capture_output=True, text=True, timeout=2.0,
                **hidden_subprocess_kwargs(),
            )
            if r.returncode == 0 and "GPU" in (r.stdout or ""):
                return "cuda", "int8_float16"
    except Exception:
        pass
    return "cpu", "int8"


class VoiceCommandListener:
    def __init__(
        self,
        *,
        backend: str = "auto",
        # Model selection for command mode. "auto" picks medium.en on
        # CUDA and small.en on CPU at construction time — both are
        # English-only models with full-depth (24/32-layer) decoders.
        # The earlier default (distil-large-v3) had only a 2-layer
        # decoder which cannot properly attend to the initial_prompt
        # on short 1-3s utterances, producing rhythm-correct but
        # phoneme-wrong transcripts like "play spotify" → "Plots
        # Spencer". medium.en/small.en outperform distil-large-v3 on
        # every published short-English benchmark; the latency hit
        # (~100-400ms on GPU) is acceptable for command mode where
        # accuracy dominates. distil-large-v3 stays in the fallback
        # chain so users who already have it cached don't trigger a
        # re-download.
        model_name: str = "auto",
        model_fallbacks: tuple[str, ...] = ("small.en", "base.en", "distil-large-v3"),
        sample_rate: int = 48000,
        # 40ms blocks (was 80ms): finer VAD polling granularity.
        # The end-of-utterance check fires on every block, so smaller
        # blocks = sooner detection that the user stopped talking. CPU
        # cost is negligible (RMS over 1920 samples).
        block_duration: float = 0.04,
        min_voice_seconds: float = 0.32,
        min_command_seconds: float = 0.68,
        # Trailing silence to declare end-of-utterance.
        # History:
        #   0.8s  → too snappy, cut off mid-phrase commands like
        #           "open youtube on google chrome"
        #   3.0s  → tolerated mid-sentence pauses but every command
        #           felt slow — user reported 4-5 s before action.
        #   1.5s  → middle ground. Still tolerates pauses up to 1.0 s
        #           between words (combined with the loud-peak grace
        #           counter below, which holds silence_blocks at 0
        #           during the first 0.5 s after any speech peak).
        end_silence_seconds: float = 1.5,
        start_timeout_seconds: float = 5.0,
        whisper_cpp_command: tuple[str, ...] | None = None,
        whisper_cpp_model_path: Path | None = None,
        preferred_input_device: str | int | None = None,
        input_gain: float = 1.0,
        input_gain_auto: bool = True,
    ) -> None:
        if platform.system() == "Windows":
            self._available = True
        else:
            # macOS / Linux: the listener runs on the portable stack —
            # sounddevice (CoreAudio/ALSA) capture + faster-whisper transcription.
            # The Windows-only whisper.cpp .exe batch path simply isn't used here
            # (_transcribe_file falls back to faster-whisper). Gate on
            # faster-whisper being importable so we degrade cleanly if absent
            # rather than instantly returning "command not understood".
            import importlib.util

            self._available = importlib.util.find_spec("faster_whisper") is not None
        self._message = "voice idle"
        # Wall-clock time.time() captured at VAD end-of-speech for the
        # most-recent listen() call. Used downstream to anchor time-
        # sensitive commands ("clip that") to when the user actually
        # spoke instead of when the dispatcher fired.
        self._last_speech_end_ts: float | None = None
        self._backend = str(backend or "auto").strip().lower()
        # Resolve the "auto" model sentinel to a concrete model based
        # on the hardware. medium.en on CUDA (best English accuracy
        # under our latency budget); small.en on CPU (caps worst-
        # case CPU latency while still beating distil-large-v3 on
        # short English commands). The detection is cheap and only
        # runs once at construction.
        resolved_model = str(model_name or "").strip()
        if resolved_model.lower() == "auto":
            try:
                _dev, _ctype = _pick_whisper_device()
            except Exception:
                _dev = "cpu"
            resolved_model = "medium.en" if _dev == "cuda" else "small.en"
        self._model_name = resolved_model
        self._model_candidates = tuple(dict.fromkeys((resolved_model, *model_fallbacks)))
        self._sample_rate = int(sample_rate)
        self._block_duration = float(block_duration)
        self._min_voice_seconds = float(min_voice_seconds)
        self._min_command_seconds = max(float(min_command_seconds), self._min_voice_seconds)
        self._end_silence_seconds = float(end_silence_seconds)
        self._start_timeout_seconds = float(start_timeout_seconds)
        touchless_models = Path.home() / "Documents" / "TouchlessVoiceModels"
        legacy_models = Path.home() / "Documents" / "HGRVoiceModels"
        self._model_root = touchless_models if touchless_models.exists() else legacy_models
        # Kept for backwards compatibility with any callers; the real lookup
        # goes through `_candidate_whisper_roots()` so that the PyInstaller
        # bundle finds whisper-cli.exe inside the install folder instead of
        # ~/Documents/whisper.cpp (which only exists on dev machines).
        self._whisper_cpp_root = Path.home() / "Documents" / "whisper.cpp"
        self._model = None
        self._whisper_cpp_command = tuple(whisper_cpp_command or ())
        self._whisper_cpp_model_path = whisper_cpp_model_path
        self._whisper_cpp_vad_model_path: Path | None = None
        self._app_hints: tuple[str, ...] = ()
        self._preferred_input_device_name: str | None = None
        self._preferred_input_device_index: int | None = None
        self._preferred_input_device = preferred_input_device
        try:
            gain_value = float(input_gain)
        except (TypeError, ValueError):
            gain_value = 1.0
        self._input_gain = max(0.1, min(10.0, gain_value))
        self._input_gain_auto = bool(input_gain_auto)
        if isinstance(preferred_input_device, str) and preferred_input_device.strip():
            self.set_input_device_name(preferred_input_device)
        elif isinstance(preferred_input_device, int):
            self._preferred_input_device_index = preferred_input_device
        # Optional override: when set, the voice pipeline reads PCM from
        # this source instead of sounddevice. Used for the phone-camera
        # QR flow's mic stream.
        self._external_audio_source = None
        # Auto-classified mic profile (per-device tuning for trigger
        # floor, AGC target, end-silence window, etc.). Set by
        # `_apply_mic_profile_from_current_device()` whenever the mic
        # changes — including at init below. Default = generic so the
        # tuning constants below stay valid even if classification
        # fails.
        try:
            from .mic_profile import classify_mic, MicProfile, MicClass  # type: ignore
            self._mic_profile = classify_mic(None)
        except Exception:
            self._mic_profile = None
        try:
            self._apply_mic_profile_from_current_device()
        except Exception:
            pass

    def _apply_mic_profile_from_current_device(self) -> None:
        """Classify the currently-selected mic device + cache its
        per-class MicProfile on self. Called from __init__ and from
        every device-change entry point so trigger floors / AGC
        target / end-silence window track the current hardware.
        Pure best-effort — failures (no sounddevice, no name, classifier
        import broken) silently leave self._mic_profile = None and the
        listener falls back to the old hard-coded constants below."""
        try:
            from .mic_profile import classify_mic
        except Exception:
            return
        name = self._preferred_input_device_name
        sr = None
        host_api = None
        max_ch = None
        if name:
            try:
                import sounddevice as sd  # type: ignore
                for dev in sd.query_devices():
                    if str(dev.get("name", "")).strip() == name.strip():
                        sr = int(dev.get("default_samplerate") or 0) or None
                        max_ch = int(dev.get("max_input_channels") or 0) or None
                        try:
                            host_api = sd.query_hostapis(int(dev.get("hostapi", 0))).get("name")
                        except Exception:
                            host_api = None
                        break
            except Exception:
                pass
        is_phone = self._external_audio_source is not None
        try:
            self._mic_profile = classify_mic(
                name,
                sample_rate=sr,
                host_api_name=host_api,
                max_input_channels=max_ch,
                is_external_phone=is_phone,
            )
            # Auto-apply suggested_gain when the user hasn't manually
            # touched the slider — gives a Razer Kiyo Pro 3.0× out of
            # the box and a Yeti 0.6× without any user action. Manual
            # users (input_gain_auto=False) keep their explicit value.
            applied_auto_gain = False
            if self._input_gain_auto and self._mic_profile is not None:
                try:
                    self._input_gain = max(
                        0.1,
                        min(10.0, float(self._mic_profile.suggested_gain)),
                    )
                    applied_auto_gain = True
                except Exception:
                    pass
            try:
                import sys as _sys
                _sys.stderr.write(
                    f"[voice] mic profile: device={name!r} "
                    f"class={self._mic_profile.mic_class.value} "
                    f"suggested_gain={self._mic_profile.suggested_gain:.2f} "
                    f"trigger_floor={self._mic_profile.trigger_floor:.4f} "
                    f"auto_gain={'applied' if applied_auto_gain else 'skipped (manual)'}\n"
                )
                _sys.stderr.flush()
            except Exception:
                pass
        except Exception:
            self._mic_profile = None

    def set_external_audio_source(self, source) -> None:
        """Install an object that exposes `sd.InputStream.read(frames)`
        semantics — returns `(ndarray (frames,1) float32, overflow)`.

        When set, the next `_record_to_wav()` call will read from this
        source instead of opening a sounddevice input stream. Pass
        `None` to clear and go back to local mic.
        """
        self._external_audio_source = source

    @property
    def available(self) -> bool:
        return self._available

    @property
    def message(self) -> str:
        return self._message


    def set_preferred_input_device(self, device: str | int | None) -> None:
        self._preferred_input_device = device

    def list_input_devices(self) -> list[tuple[int, str]]:
        try:
            import sounddevice as sd
            devices = sd.query_devices()
        except Exception:
            return []
        results: list[tuple[int, str]] = []
        for index, device in enumerate(devices):
            try:
                max_input = int(device.get("max_input_channels", 0))
            except Exception:
                max_input = 0
            if max_input <= 0:
                continue
            name = str(device.get("name") or f"Input Device {index}").strip()
            results.append((index, name))
        return results

    def set_input_device_name(self, device_name: str | None) -> None:
        normalized = str(device_name or "").strip() or None
        self._preferred_input_device_name = normalized
        self._preferred_input_device_index = self._resolve_input_device_index(normalized)
        # Re-classify the new mic so trigger floors / AGC target / etc.
        # track the per-class profile.
        try:
            self._apply_mic_profile_from_current_device()
        except Exception:
            pass

    def input_device_name(self) -> str | None:
        return self._preferred_input_device_name

    def input_device_index(self) -> int | None:
        """The sounddevice (PortAudio) input-device index the listener
        is currently using. Returns None when the listener never
        resolved a mic, or when the saved name no longer matches any
        WASAPI input (device unplugged). Used by the clip-cache audio
        capture so it can reuse the listener's already-resolved mic
        without going through the DirectShow naming layer."""
        return self._preferred_input_device_index

    def set_input_gain(self, gain: float) -> None:
        try:
            value = float(gain)
        except (TypeError, ValueError):
            value = 1.0
        self._input_gain = max(0.1, min(10.0, value))
        # User touched the slider → switch to manual mode so the next
        # device-change doesn't silently override their tuning.
        self._input_gain_auto = False

    def set_input_gain_auto(self, auto: bool) -> None:
        """Toggle auto-mode for the input gain. When True, the next
        mic-change re-applies the suggested_gain from the profile;
        when False, the user's explicit `_input_gain` wins."""
        self._input_gain_auto = bool(auto)

    @property
    def input_gain(self) -> float:
        return self._input_gain

    def _resolve_input_device_index(self, device_name: str | None) -> int | None:
        if not device_name:
            return None
        try:
            import sounddevice as sd
            import platform
        except Exception:
            return None
        try:
            devices = sd.query_devices()
        except Exception:
            return None
        # Mirror the WASAPI filter used by list_input_microphones at
        # module top — without it, a saved mic name can resolve to the
        # MME or DirectSound duplicate which has worse drivers and
        # different latency. The dropdown is WASAPI-only on Windows
        # so resolution must be too.
        wasapi_index: int | None = None
        try:
            if platform.system() == "Windows":
                for idx, host in enumerate(sd.query_hostapis()):
                    host_name = str(host.get("name", "") or "").strip().lower()
                    if host_name == "windows wasapi":
                        wasapi_index = idx
                        break
        except Exception:
            wasapi_index = None
        # First pass: WASAPI-only on Windows.
        for index, device in enumerate(devices):
            try:
                max_inputs = int(device.get("max_input_channels", 0) or 0)
            except Exception:
                max_inputs = 0
            if max_inputs <= 0:
                continue
            if wasapi_index is not None:
                try:
                    if int(device.get("hostapi", -1)) != wasapi_index:
                        continue
                except Exception:
                    continue
            name = str(device.get("name", "") or "").strip()
            if name == device_name:
                return int(index)
        # Defensive fallback: if the WASAPI pass missed, retry across
        # all host APIs (legacy configs from before the dropdown was
        # WASAPI-filtered, or PortAudio builds without WASAPI).
        for index, device in enumerate(devices):
            try:
                max_inputs = int(device.get("max_input_channels", 0) or 0)
            except Exception:
                max_inputs = 0
            if max_inputs <= 0:
                continue
            name = str(device.get("name", "") or "").strip()
            if name == device_name:
                return int(index)
        return None

    def _preserve_debug_audio(self, audio_path: Path, *, transcript_mode: str, text: str) -> None:
        try:
            debug_dir = Path.home() / ".touchless" / "voice_debug"
            debug_dir.mkdir(parents=True, exist_ok=True)
            import time
            stamp = time.strftime("%Y%m%d_%H%M%S")
            snippet = re.sub(r"[^a-zA-Z0-9]+", "_", str(text or "empty"))[:40] or "empty"
            target = debug_dir / f"{stamp}_{transcript_mode}_{snippet}.wav"
            shutil.move(str(audio_path), str(target))
        except Exception:
            try:
                audio_path.unlink(missing_ok=True)
            except Exception:
                pass

    def set_app_hints(self, app_names: Iterable[str]) -> None:
        hints: list[str] = []
        for item in app_names:
            normalized = self._normalize_hint_name(str(item or ""))
            if normalized:
                hints.append(normalized)
        self._app_hints = tuple(dict.fromkeys(hints))[:32]

    def prewarm(self) -> None:
        if not self._available:
            return
        try:
            if self._whisper_cpp_ready():
                self._message = "voice ready: whisper.cpp"
                return
        except Exception:
            pass
        try:
            model = self._ensure_model()
            self._message = f"voice ready: {self._model_name}"
            # Trigger a real decode so CUDA graphs + cuDNN
            # convolutions actually materialize. Without this, the
            # FIRST live command pays 800-1500ms of warmup on the
            # critical path — exactly what makes voice commands feel
            # sluggish on cold start. The dummy decode runs on a
            # 1.5s buffer of silence which whisper resolves in
            # ~100-300ms; faster than waiting for it on the first
            # real command.
            try:
                import tempfile
                import numpy as _np
                from pathlib import Path as _Path
                import wave as _wave
                tmp = _Path(tempfile.gettempdir()) / "touchless_prewarm.wav"
                sr = 16000
                silence = _np.zeros(int(sr * 1.5), dtype=_np.float32)
                # Write as 16-bit PCM mono WAV (whisper accepts).
                pcm16 = (silence * 32767.0).astype(_np.int16).tobytes()
                with _wave.open(str(tmp), "wb") as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(sr)
                    wf.writeframes(pcm16)
                if model is not None:
                    try:
                        list(model.transcribe(
                            str(tmp),
                            language="en",
                            beam_size=1,
                            best_of=1,
                            temperature=0.0,
                            vad_filter=False,
                            condition_on_previous_text=False,
                            initial_prompt=None,
                        )[0])
                    except Exception:
                        pass
                try:
                    tmp.unlink(missing_ok=True)
                except Exception:
                    pass
            except Exception:
                pass
        except Exception:
            pass

    def listen(
        self,
        *,
        max_seconds: float = 15.0,
        status_callback: Callable[[str], None] | None = None,
        stop_event=None,
        transcript_mode: str = "command",
    ) -> VoiceCommandResult:
        # Reset for this listen() call so a stale value from the prior
        # call (or one captured during the current call but unrelated
        # to a successful transcription) can't leak into downstream
        # time-anchored handlers.
        self._last_speech_end_ts = None
        if not self._available:
            self._message = "voice unavailable on this platform"
            return VoiceCommandResult(heard_text="", success=False, message=self._message)

        transcript_mode_raw = str(transcript_mode or "").strip().lower()
        if transcript_mode_raw == "dictation":
            transcript_mode = "dictation"
        elif transcript_mode_raw in {"save_prompt", "save_location", "save_destination"}:
            transcript_mode = "save_prompt"
        elif transcript_mode_raw in {"playlist", "playlist_name", "playlist_prompt"}:
            transcript_mode = "playlist"
        else:
            transcript_mode = "command"

        try:
            if status_callback is not None:
                status_callback("listening")
            audio_path = self._record_to_wav(
                max_seconds=max_seconds,
                stop_event=stop_event,
                transcript_mode=transcript_mode,
            )
        except Exception as exc:
            self._message = f"voice capture failed: {type(exc).__name__}"
            return VoiceCommandResult(heard_text="", success=False, message=self._message)

        if audio_path is None:
            if transcript_mode == "dictation":
                self._message = "dictation paused" if stop_event is not None and stop_event.is_set() else "dictation waiting..."
            else:
                self._message = "voice command not heard"
            return VoiceCommandResult(heard_text="", success=False, message=self._message)

        text = ""
        transcription_failed = False
        try:
            if status_callback is not None:
                status_callback("recognizing")
            text = self._transcribe_file(audio_path, transcript_mode=transcript_mode)
            # Phase-3 wiring: post-process through TranscriptionRouter.
            # When the fast transcript contains an email / URL / file
            # path / digit run / destructive verb, re-decode with the
            # accurate backend before returning. Cheap when not needed.
            if text and audio_path.exists():
                text = self._maybe_escalate_transcription(
                    fast_text=text, audio_path=audio_path,
                    audio_seconds=float(max_seconds or 0.0),
                    transcript_mode=transcript_mode)
        except Exception:
            # v1.1.7: SAPI PowerShell fallback removed (Windows
            # Defender flagged its `-EncodedCommand` shell-out as a
            # malware-dropper pattern). Whisper failing here is now a
            # terminal condition — the user sees "voice transcription
            # failed" and the audio is preserved for debug. Whisper
            # ships with three backend variants (CUDA/Vulkan/CPU) so
            # a total transcription-engine failure only happens on a
            # broken install; a rare miss is preferable to shipping a
            # binary that gets quarantined by every scanner.
            transcription_failed = True
            self._message = "voice transcription failed"
            self._preserve_debug_audio(audio_path, transcript_mode=transcript_mode, text="<transcription_failed>")
            return VoiceCommandResult(heard_text="", success=False, message=self._message)
        finally:
            if not transcription_failed:
                if text:
                    try:
                        audio_path.unlink(missing_ok=True)
                    except Exception:
                        pass
                else:
                    self._preserve_debug_audio(audio_path, transcript_mode=transcript_mode, text="empty")

        if not text:
            # v1.1.7: SAPI PowerShell fallback removed. See the earlier
            # except-branch comment for rationale. When Whisper produces
            # an empty transcript, surface that directly.
            self._message = "dictation not understood" if transcript_mode == "dictation" else "voice command not understood"
            return VoiceCommandResult(heard_text="", success=False, message=self._message)

        self._message = f"heard: {text}"
        return VoiceCommandResult(
            heard_text=text,
            success=True,
            message=self._message,
            speech_end_ts=self._last_speech_end_ts,
        )

    def _record_to_wav(
        self,
        *,
        max_seconds: float,
        stop_event=None,
        transcript_mode: str = "command",
    ) -> Path | None:
        import sounddevice as sd

        transcript_mode_raw = str(transcript_mode or "").strip().lower()
        if transcript_mode_raw == "dictation":
            transcript_mode = "dictation"
        elif transcript_mode_raw in {"save_prompt", "save_location", "save_destination"}:
            transcript_mode = "save_prompt"
        elif transcript_mode_raw in {"playlist", "playlist_name", "playlist_prompt"}:
            transcript_mode = "playlist"
        else:
            transcript_mode = "command"
        sample_rate = self._sample_rate
        block_size = max(256, int(sample_rate * self._block_duration))
        max_blocks = max(1, int(max_seconds / self._block_duration))
        ambient_blocks = max(3, int(0.6 / self._block_duration))
        if transcript_mode == "dictation":
            start_timeout_seconds = self._start_timeout_seconds + 1.8
            min_active_seconds = max(self._min_voice_seconds, 0.60)
        elif transcript_mode == "playlist":
            start_timeout_seconds = max(1.1, self._start_timeout_seconds - 0.20)
            min_active_seconds = max(self._min_voice_seconds, 0.40)
        else:
            start_timeout_seconds = self._start_timeout_seconds
            min_active_seconds = self._min_command_seconds
        start_timeout_blocks = max(1, int(start_timeout_seconds / self._block_duration))

        voice_started = False
        voice_blocks = 0
        silence_blocks = 0
        # Count of consecutive blocks where rms was BELOW trigger_threshold
        # AFTER voice_started. This is the "time since last loud peak"
        # counter — silence_blocks++ ONLY accumulates after this counter
        # exceeds SILENCE_GRACE_BLOCKS, so natural inter-word gaps (which
        # are below silence_threshold for quiet mics) don't get
        # mis-counted as end-of-utterance silence. A quiet mic where
        # speech RMS is ~0.005-0.008 produces sub-trigger blocks during
        # every consonant + inter-word gap; this counter prevents those
        # from accumulating to 75 (3 s) and ending VAD mid-phrase.
        blocks_since_trigger_peak = 0
        # 0.5 second of grace — still tolerates inter-word gaps for
        # quiet mics (consonants + word boundaries) but no longer adds
        # a full extra second of dead air to every command. Combined
        # with the shorter end_silence_seconds=1.5 above, total
        # post-speech wait drops from ~4.0 s to ~2.0 s on a typical
        # command. Speed-up: ~2 s per command, measurable as snappier
        # dispatch.
        SILENCE_GRACE_BLOCKS = max(1, int(0.5 / self._block_duration))
        ambient_levels: list[float] = []
        chunks: list[np.ndarray] = []
        # Preroll: how much pre-trigger audio to prepend when VAD
        # finally trips. Bumped 0.60 → 1.20 → 2.00s — users still
        # reported intermittent "clip that" → "that" at 1.20s. The
        # soft K at the start of "clip" rides below the energy-based
        # trigger threshold, so VAD doesn't fire until mid-word. A
        # 2-second preroll buffers two full syllable durations so even
        # a slow-onset utterance has its first phoneme captured. Cost
        # is small: extra 0.8s of ambient audio prepended to each
        # command, which whisper trims cleanly.
        preroll_blocks = max(4, int(2.00 / self._block_duration))
        preroll: deque[np.ndarray] = deque(maxlen=preroll_blocks)
        max_rms_seen = 0.0
        final_noise_floor = 0.0
        final_trigger_threshold = 0.0

        # Stream config matches the mic test exactly. The mic test
        # at main_window.py _start_mic_test uses callback-driven
        # capture with these same kwargs and produces clean audio
        # on the Kiyo Pro — we mirror it. PortAudio's read() vs
        # callback paths go through different code on WASAPI; the
        # callback path is event-driven (clean), read is polling-
        # mode (flaky on some drivers).
        stream_kwargs = {
            "samplerate": sample_rate,
            "channels": 1,
            "dtype": "float32",
            "blocksize": 0,
        }
        if self._preferred_input_device_index is not None:
            stream_kwargs["device"] = self._preferred_input_device_index

        # Route through the phone-posted PCM source instead of
        # sounddevice when one has been installed. The external source
        # exposes the same read(frames) -> (ndarray (frames,1) float32,
        # overflow) contract that sd.InputStream.read returns, so no
        # changes to the loop below are needed.
        external = self._external_audio_source

        # NOTE: A pre-open block here used to query the device's
        # default_samplerate and REASSIGN sample_rate to it. That
        # logic was a bug on UVC webcam mics where the query
        # returns one rate but the driver delivers another.
        # Removed; we now match the mic test's hardcoded approach.

        # Callback-driven capture buffer. The mic test uses callback
        # mode and produces clean audio on the Kiyo Pro; my prior
        # read()-polling loop produced corrupted "low robotic"
        # WAVs. PortAudio's WASAPI shared-mode polling path is
        # known-flaky on some UVC drivers; the callback path is
        # event-driven and goes through a different (clean) code
        # path inside PortAudio.
        import threading as _threading
        _audio_pieces: list = []
        _audio_pieces_lock = _threading.Lock()
        _first_shape_logged = [False]  # mutable closure cell

        if external is None:
            def _audio_callback(indata, frames, time_info, status):
                if indata is None:
                    return
                try:
                    arr = np.asarray(indata, dtype=np.float32)
                    if not _first_shape_logged[0]:
                        _first_shape_logged[0] = True
                        try:
                            import sys as _sys
                            _sys.stderr.write(
                                f"[voice] callback: first piece "
                                f"shape={arr.shape} dtype={arr.dtype} "
                                f"frames_param={frames}\n"
                            )
                            _sys.stderr.flush()
                        except Exception:
                            pass
                    if arr.ndim > 1:
                        mono = np.ascontiguousarray(arr[:, 0])
                    elif arr.size == frames * 2:
                        # 1D interleaved stereo — take every other.
                        # Some drivers do this even when channels=1
                        # is requested. Hitting this branch IS the
                        # smoking gun for "low robotic" symptom.
                        try:
                            import sys as _sys
                            _sys.stderr.write(
                                f"[voice] callback: 1D interleaved "
                                f"stereo detected — taking [::2]\n"
                            )
                            _sys.stderr.flush()
                        except Exception:
                            pass
                        mono = np.ascontiguousarray(arr[::2])
                    else:
                        mono = arr
                    with _audio_pieces_lock:
                        _audio_pieces.append(mono.copy())
                except Exception:
                    pass
            stream_kwargs["callback"] = _audio_callback

        if external is not None and hasattr(external, "drain"):
            # Discard any audio buffered before this session started —
            # otherwise a tap sound or idle chatter from seconds ago
            # fires voice activation on the very first read and the
            # loop enters the recording path with no real speech.
            external.drain()
        if external is not None:
            stream_ctx = external
        else:
            stream_ctx = sd.InputStream(**stream_kwargs)
            # NOTE: No post-open rate renegotiation. The mic test
            # in main_window.py never reads stream.samplerate
            # back; it just uses the same 48000 it requested.
            # We do the same — trying to be "smart" about the
            # actual rate is what produced the bug above.

        # One-shot diagnostic: warn (in the log only — don't bother
        # the user mid-command) when the actual device sample rate
        # looks like Bluetooth HFP mode (8 / 16 kHz). HFP is the
        # narrow-band call profile most BT headsets use when their
        # mic is active; whisper accuracy collapses below 16 kHz and
        # users blame Iris for what's a Windows audio routing
        # problem. Logged once per session via _bt_warning_logged.
        if not getattr(self, "_bt_warning_logged", False):
            try:
                actual_rate = sample_rate
                if external is None:
                    info = sd.query_devices(
                        stream_kwargs.get("device"), "input")
                    actual_rate = int(info.get("default_samplerate")
                                      or sample_rate)
                if actual_rate and actual_rate < 16000:
                    import sys as _sys
                    print(
                        f"[voice] WARNING input device default rate is "
                        f"{actual_rate} Hz — looks like Bluetooth HFP "
                        "(narrow-band call mode). Whisper accuracy drops "
                        "sharply below 16 kHz. Fix: in Windows Sound "
                        "Settings switch your headset's INPUT to a wired "
                        "mic, or pair the headset as 'Hands-free + A2DP "
                        "Stereo' (mic disabled) and use the laptop mic.",
                        file=_sys.stderr, flush=True)
                self._bt_warning_logged = True
            except Exception:
                self._bt_warning_logged = True  # don't retry every block

        # Track which branch exited the capture loop so the chunks=N
        # log can be classified as vad_end / timeout_cap / stop_event /
        # start_timeout. Future runaways are then diagnosable from the
        # log alone instead of guessing.
        vad_end_reached = False
        stop_event_fired = False
        start_timeout_fired = False
        # Pre-initialize so the log can't NameError if the loop body
        # never reached the threshold computation (e.g., max_blocks=0
        # in pathological config).
        silence_threshold = 0.0
        # Accumulator for assembling block_size-sized frames from
        # the variable-sized callback pieces (callback path) OR
        # from external.read() (phone path).
        _audio_accum = np.empty(0, dtype=np.float32)
        import time as _time_mod
        import sys as _vsys
        # macOS hang diagnostic: pinpoint whether an intermittent "listens
        # forever" is the CoreAudio stream __enter__ blocking vs the callback
        # never delivering. If only the first line prints, the stream start
        # hung; if both print but no "callback: first piece", the callback
        # never fired.
        print(f"[voice] opening input stream rate={sample_rate} device={stream_kwargs.get('device')} external={external is not None}", file=_vsys.stderr, flush=True)
        with stream_ctx as stream:
            print("[voice] input stream open — entering capture loop", file=_vsys.stderr, flush=True)
            for block_index in range(max_blocks):
                if stop_event is not None and stop_event.is_set():
                    stop_event_fired = True
                    if voice_started and chunks:
                        break
                    return None
                if external is not None:
                    # External (phone) source still uses read().
                    data, _overflow = stream.read(block_size)
                    arr = np.asarray(data, dtype=np.float32)
                    if arr.ndim > 1:
                        mono = np.ascontiguousarray(arr[:, 0])
                    else:
                        mono = arr
                else:
                    # sd.InputStream callback path: pull pieces from
                    # _audio_pieces into _audio_accum until we have
                    # enough for one block_size VAD window.
                    _wait_start = _time_mod.monotonic()
                    while len(_audio_accum) < block_size:
                        new_pieces = None
                        with _audio_pieces_lock:
                            if _audio_pieces:
                                new_pieces = list(_audio_pieces)
                                _audio_pieces.clear()
                        if new_pieces:
                            _audio_accum = np.concatenate(
                                [_audio_accum, *new_pieces]
                            )
                            continue
                        if _time_mod.monotonic() - _wait_start > 2.0:
                            break
                        _time_mod.sleep(0.002)
                    if len(_audio_accum) < block_size:
                        # Driver stopped delivering — bail.
                        break
                    mono = _audio_accum[:block_size]
                    _audio_accum = _audio_accum[block_size:]
                if mono.ndim == 0:
                    mono = np.asarray([float(mono)], dtype=np.float32)
                # Phone audio gets a modest 4x boost in the phone-side
                # worklet (just enough to lift iOS's extremely quiet
                # getUserMedia baseline). The user's PC-side gain then
                # brings it to transcription-ready levels — typical
                # settings of 5-8x give ~20-32x total, matching what
                # the in-app mic test uses for clear phone playback.
                # Mild clipping from the peak normalization step is
                # fine; whisper handles it. Earlier clipping issues
                # were from a 20x phone boost, not the PC gain.
                gain = self._input_gain
                if gain != 1.0:
                    mono = mono * gain
                rms = float(np.sqrt(np.mean(np.square(mono))) + 1e-9)

                if not voice_started and block_index < ambient_blocks:
                    ambient_levels.append(rms)
                elif not voice_started and rms < max(0.003, ambient_levels[-1] * 1.6 if ambient_levels else 0.0):
                    # Rolling refresh: outside the initial 600 ms window,
                    # keep folding in CURRENT quiet samples so the noise
                    # floor tracks fan-on / AC-on / a stream starting
                    # mid-session. Cap ambient_levels at 40 entries so
                    # the median stays responsive to recent changes
                    # without flapping. Only adds samples that are
                    # clearly NOT voice (under 1.6× the recent floor).
                    ambient_levels.append(rms)
                    if len(ambient_levels) > 40:
                        ambient_levels = ambient_levels[-40:]

                noise_floor = self._estimate_noise_floor(ambient_levels)
                # Lowered the absolute minimum trigger from 0.008 to
                # 0.005 — at the higher value, quiet mics (Razer
                # Kiyo Pro webcam at gain 0.48, etc.) whose VOICE
                # RMS sits around 0.005-0.006 never tripped VAD even
                # though signal-to-noise was healthy (1.6x noise
                # floor). The dynamic `noise_floor * 2.0` term is
                # still the dominant signal in quiet rooms (raises
                # the trigger above ambient noise spikes), so the
                # lower floor only kicks in for genuinely quiet
                # mics and doesn't add false triggers.
                # Per-mic-class floors when a profile is available
                # (lower for webcams, higher for hot condensers, etc.).
                # Falls back to the universal 0.005 / 0.003 if the
                # classifier was unavailable at init.
                _t_floor = 0.005
                _s_floor = 0.003
                if self._mic_profile is not None:
                    try:
                        _t_floor = float(self._mic_profile.trigger_floor)
                        _s_floor = float(self._mic_profile.silence_floor)
                    except Exception:
                        pass
                trigger_threshold = max(noise_floor * 2.0, _t_floor)
                # Silence threshold widened from 1.3x to 1.8x noise_floor
                # with an absolute 0.006 floor. The earlier 1.3x margin
                # was so tight (only ~30% above ambient) that any
                # micro-burst — breath, mouse click, fan ripple, voice
                # tail decay — sat above silence_threshold and zeroed
                # the silence counter. End-of-speech never accumulated
                # the 20 consecutive sub-threshold blocks (0.8s) it
                # needed to fire, the loop ran to the 12s safety cap,
                # and the user perceived "command listened 12s extra".
                # 1.8x with an absolute 0.006 floor leaves a comfortable
                # gap below the 2.0x trigger so a returning whisper of
                # the same level doesn't accidentally re-activate VAD.
                # Dropped multiplier 3.0 → 2.0 and the absolute floor
                # 0.006 → 0.0035. With the quieter mic the user is now
                # running, the 0.006 floor put 'last five minutes'-class
                # trailing-soft-speech BELOW the silence bar, so VAD
                # treated 'clip last five minutes' as 'clip <silence>'
                # and Whisper only ever heard 'Clip'.
                # The lower bar (~0.0035-0.005 for typical Kiyo Pro
                # noise floors) sits above ambient but BELOW even
                # whispered-trailing-syllable RMS, so quiet continued
                # speech keeps the silence counter at 0. The hysteresis
                # below absorbs single-block ambient blips.
                silence_threshold = max(noise_floor * 2.0, _s_floor, 0.0035)
                final_noise_floor = noise_floor
                final_trigger_threshold = trigger_threshold
                if rms > max_rms_seen:
                    max_rms_seen = rms

                if voice_started:
                    chunks.append(mono.copy())
                    voice_blocks += 1
                    # SILENCE COUNTING WITH LOUD-PEAK GRACE.
                    #
                    # Problem this solves: the user's mic captures
                    # speech at ~0.005-0.008 peak RMS today. With
                    # silence_threshold tuned anywhere realistic
                    # (~0.005), the *quiet* portions of a phrase
                    # (consonants, inter-word gaps, trailing
                    # syllables) all fall BELOW the silence bar, so
                    # the old "silence_blocks++ whenever rms <
                    # threshold" logic accumulated 75 blocks (3 s)
                    # within seconds — VAD fired end-of-utterance
                    # in the middle of "open youtube on google
                    # chrome", capturing only "Open".
                    #
                    # New logic: silence_blocks ONLY accumulates
                    # AFTER we've seen no trigger-crossing peak for
                    # SILENCE_GRACE_BLOCKS (1 s). Any rms above
                    # trigger_threshold resets the grace counter
                    # AND zeroes silence_blocks. Quiet inter-word
                    # gaps don't accumulate — they're protected by
                    # the grace window. Real "user stopped" only
                    # fires after 1 s grace + 3 s genuine silence
                    # = 4 s total end-of-speech latency, which
                    # matches the user's "3 s pause tolerance"
                    # rule once we account for the grace.
                    if rms > trigger_threshold:
                        blocks_since_trigger_peak = 0
                        silence_blocks = 0
                    else:
                        blocks_since_trigger_peak += 1
                        if blocks_since_trigger_peak > SILENCE_GRACE_BLOCKS:
                            if rms <= silence_threshold:
                                silence_blocks += 1
                            elif rms <= silence_threshold * 2.0:
                                silence_blocks = max(0, silence_blocks - 1)
                            else:
                                silence_blocks = 0
                        # else: still in grace window, freeze the
                        # silence counter wherever it is (don't add,
                        # don't subtract).
                    active_seconds = voice_blocks * self._block_duration
                    required_silence = self._adaptive_end_silence_seconds(active_seconds, transcript_mode=transcript_mode)
                    # Local sounddevice mic: tighter end-silence
                    # window — 3s of dead air after a command felt
                    # too long to users. Phone source keeps the
                    # default longer window because phone's noise
                    # gate produces aggressive zeros between words
                    # and a tighter cutoff would chop sentences
                    # mid-phrase. Heuristic: external sources keep
                    # configured end_silence_seconds (3.0); local
                    # sources cut it ~33% to ~2.0s for command
                    # mode and similarly for save_prompt.
                    # Per-source end-of-utterance window. Local sounddevice
                    # mic: snappy. Phone source (external != None): more
                    # generous because phone noise gates inject zero
                    # samples between words and the tighter cutoff would
                    # chop sentences. Was 2.0s/3.0s; tightened in tandem
                    # with the default drop to 0.8s (see __init__ comment).
                    if external is None and transcript_mode in {"command", "save_prompt"}:
                        # 3.0s end-of-speech window. The user wants
                        # room for mid-command pauses ("play... uh
                        # ... spotify"). Once they start speaking,
                        # 3 full seconds of dead air are required
                        # before VAD declares end-of-utterance.
                        required_silence = min(required_silence, 3.0)
                    elif external is not None:
                        # Phone / remote sources keep a softer 1.2s.
                        required_silence = max(required_silence, 1.2)
                    if (
                        active_seconds >= min_active_seconds
                        and silence_blocks * self._block_duration >= required_silence
                    ):
                        # Mark when speech actually ended — used by
                        # time-anchored commands like "clip that" so
                        # the clip window ends at this moment instead
                        # of the much-later moment when execute() runs
                        # (after whisper + parse + dispatch). Wall-clock
                        # (time.time()) matches the clip-cache segment
                        # timestamps so the right edge lines up cleanly.
                        self._last_speech_end_ts = time.time()
                        vad_end_reached = True
                        try:
                            import sys as _sys
                            _sys.stderr.write(
                                f"[clip-anchor] listener VAD-end speech_end_ts={self._last_speech_end_ts:.3f} "
                                f"active_seconds={active_seconds:.2f} silence_seconds={silence_blocks * self._block_duration:.2f}\n"
                            )
                            _sys.stderr.flush()
                        except Exception:
                            pass
                        break
                elif rms >= trigger_threshold:
                    voice_started = True
                    if preroll:
                        # Cap the preroll dumped into chunks to the
                        # last ~320ms (8 blocks at 40ms). The full
                        # 2s preroll buffer existed for VAD recovery
                        # at activation time, but feeding all 2s of
                        # pre-speech ambient to Whisper made the WAV
                        # >50% silence. Whisper's decoder then
                        # defaults to high-frequency training-data
                        # closing phrases ("Thanks for watching",
                        # "Mhm", "Please subscribe") because the
                        # silence-heavy buffer looks like end-of-
                        # video residual. 320ms is enough to catch
                        # the /p/ /t/ /k/ closure before the burst
                        # without flooding Whisper with silence.
                        MAX_PREROLL_FOR_WHISPER = 8
                        kept = list(preroll)[-MAX_PREROLL_FOR_WHISPER:]
                        chunks.extend(kept)
                        voice_blocks = len(kept)
                        preroll.clear()
                    else:
                        voice_blocks = 0
                    chunks.append(mono.copy())
                    voice_blocks += 1
                    silence_blocks = 0
                elif block_index + 1 >= start_timeout_blocks:
                    start_timeout_fired = True
                    break
                else:
                    preroll.append(mono.copy())

        import sys as _sys
        source_desc = (
            "phone-external"
            if external is not None
            else f"local-sd(idx={self._preferred_input_device_index} name={self._preferred_input_device_name!r})"
        )
        if vad_end_reached:
            exit_reason = "vad_end"
        elif stop_event_fired:
            exit_reason = "stop_event"
        elif start_timeout_fired:
            exit_reason = "start_timeout"
        else:
            exit_reason = "timeout_cap"
        print(
            f"[voice] mode={transcript_mode} source={source_desc} gain={self._input_gain:.2f} "
            f"model={self._model_name!r} rate={int(sample_rate)} "
            f"noise_floor={final_noise_floor:.4f} trigger={final_trigger_threshold:.4f} "
            f"silence_thr={silence_threshold:.4f} silence_blocks={silence_blocks} "
            f"max_rms_seen={max_rms_seen:.4f} voice_started={voice_started} "
            f"chunks={len(chunks)} exit={exit_reason}",
            file=_sys.stderr,
            flush=True,
        )

        if not voice_started or not chunks:
            return None

        # Trim trailing silence before WAV encoding. VAD waits for
        # required_silence (0.8s = 20 blocks at 40ms) of dead air
        # before declaring end-of-speech, but Whisper doesn't need
        # 0.8s of trailing silence — and the extra silence pushes
        # the speech-to-silence ratio of the buffer below 50%,
        # triggering training-data hallucinations ("Thanks for
        # watching."). Keep only the last ~240ms (6 blocks) of
        # silence — enough decoder context, not enough to confuse.
        # Only trim if VAD-end fired, because stop_event /
        # timeout_cap exits don't reliably end in pure silence.
        if vad_end_reached and silence_blocks > 6:
            trim_count = silence_blocks - 6
            if trim_count < len(chunks):
                chunks = chunks[: len(chunks) - trim_count]

        audio = np.concatenate(chunks).astype(np.float32, copy=False)
        # ---- preprocessing chain ------------------------------------
        # Three cheap-but-effective steps to give whisper a cleaner
        # signal. Order matters: DC first (removes any constant offset
        # the mic adds), high-pass next (kills sub-speech rumble that
        # would otherwise feed into AGC's target-RMS calc), THEN soft
        # AGC (brings quiet talkers up to a level whisper handles
        # well). All in-place numpy ops — ~1ms for a 3s buffer.
        audio = _preprocess_audio_for_whisper(
            audio, sample_rate, user_gain=self._input_gain,
        )
        peak = float(np.max(np.abs(audio))) if audio.size else 0.0
        if peak <= 0.0025:
            return None
        # Detect input clipping AND auto-attenuate gain for the NEXT
        # capture. The earlier logic only warned in stderr — useless
        # for the user whose voice command just got garbled because of
        # clipping. Now: when peak > 0.95 OR max_rms_seen > 0.85, drop
        # the running gain in half (clamped at 0.2 minimum) so the
        # next command captures cleaner audio. Also unset auto-mode
        # so a profile change can't silently bump it back up. This is
        # the safety net for users whose Windows input level is
        # already cranked and a profile auto-gain pushed them over
        # the edge — exactly what happened on the Kiyo Pro at 3.0×.
        if max_rms_seen >= 0.85 or peak >= 0.995:
            try:
                old_gain = float(self._input_gain)
                new_gain = max(0.2, old_gain * 0.5)
                if new_gain < old_gain - 1e-3:
                    self._input_gain = new_gain
                    self._input_gain_auto = False
                    import sys as _sys
                    _sys.stderr.write(
                        f"[voice] AUTO-ATTENUATE: clipping detected "
                        f"(peak={peak:.2f} max_rms={max_rms_seen:.2f}); "
                        f"cut gain {old_gain:.2f} -> {new_gain:.2f}. "
                        f"Repeat this command — should sound cleaner.\n"
                    )
                    _sys.stderr.flush()
            except Exception:
                pass
            import sys as _sys
            print(
                f"[voice] WARNING input likely clipping (max_rms={max_rms_seen:.3f} "
                f"peak={peak:.3f}). Auto-attenuating gain for the next attempt. "
                f"If clipping continues, lower the device's input volume in "
                f"Windows Sound Settings OR drop Touchless 'Mic input gain' "
                f"slider further.",
                file=_sys.stderr,
                flush=True,
            )
        # Pre-normalize-peak gain reduction when severe clipping is
        # detected: scale down by 0.7 BEFORE peak-normalise. This
        # gives whisper a less-distorted (but quieter) signal which
        # is usually still transcribable, instead of a peak-loud but
        # clipped one which usually isn't.
        if max_rms_seen >= 0.95:
            audio = audio * 0.7
            peak = float(np.max(np.abs(audio))) if audio.size else 0.0
            if peak <= 0.0025:
                return None
        # 0.85 instead of 0.95 — user reported the WAV playback was
        # "a little loud". 0.85 leaves comfortable headroom; whisper
        # is amplitude-invariant via log-mel so accuracy is unaffected.
        target_peak = 0.85
        if peak > 0.0:
            audio = audio * (target_peak / peak)

        # Diagnostic: save a debug copy of the audio at the CAPTURE
        # rate (pre-resample). If `last_command.wav` (the 16k Whisper
        # input) sounds glitchy/pitched but `last_command_raw.wav`
        # sounds clean, the bug is in our resample. If both sound
        # bad, the bug is in capture / preprocessing / sample rate.
        try:
            debug_dir = Path.home() / "Documents" / "TouchlessVoiceModels"
            debug_dir.mkdir(parents=True, exist_ok=True)
            raw_path = debug_dir / "last_command_raw.wav"
            with wave.open(str(raw_path), "wb") as wraw:
                wraw.setnchannels(1)
                wraw.setsampwidth(2)
                wraw.setframerate(int(sample_rate))
                raw_data = np.clip(audio * 32767.0, -32768.0, 32767.0).astype(np.int16)
                wraw.writeframes(raw_data.tobytes())
            import sys as _sys
            _sys.stderr.write(
                f"[voice] debug raw WAV saved to {raw_path} "
                f"(rate={int(sample_rate)} Hz)\n"
            )
            _sys.stderr.flush()
        except Exception:
            pass

        # Down-sample to whisper's native 16 kHz before encoding. Cuts
        # the WAV file 3x in size AND removes whisper's internal
        # resampler from the latency budget. Result: faster decode +
        # less variability run-to-run.
        encode_rate = 16000
        audio_resampled = _resample_to_16k(audio, sample_rate)
        with tempfile.NamedTemporaryFile(prefix="hgr_voice_", suffix=".wav", delete=False) as tmp:
            path = Path(tmp.name)
        with wave.open(str(path), "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(encode_rate)
            wav_data = np.clip(audio_resampled * 32767.0, -32768.0, 32767.0).astype(np.int16)
            wav_file.writeframes(wav_data.tobytes())
        # Debug copy: also save to ~/Documents/TouchlessVoiceModels/
        # last_command.wav so the user can listen to the EXACT audio
        # Whisper received and confirm whether the issue is
        # capture/preprocessing (audio sounds bad to a human) or
        # model/prompt (audio sounds clear but Whisper still fails).
        # This is the missing diagnostic — we've been guessing about
        # audio quality from log-derived stats instead of listening.
        try:
            debug_dir = Path.home() / "Documents" / "TouchlessVoiceModels"
            debug_dir.mkdir(parents=True, exist_ok=True)
            debug_path = debug_dir / "last_command.wav"
            shutil.copy2(str(path), str(debug_path))
            import sys as _sys
            _sys.stderr.write(
                f"[voice] debug WAV saved to {debug_path}\n"
            )
            _sys.stderr.flush()
        except Exception:
            pass
        return path

    def _adaptive_end_silence_seconds(self, active_seconds: float, *, transcript_mode: str = "command") -> float:
        if transcript_mode == "command":
            return self._end_silence_seconds
        if transcript_mode == "playlist":
            required = max(0.36, self._end_silence_seconds - 0.45)
        else:
            required = max(0.80, self._end_silence_seconds - 0.60)
        if active_seconds >= 6.0:
            required += 0.18
        return required

    def _transcribe_file(self, audio_path: Path, *, transcript_mode: str = "command") -> str:
        errors: list[str] = []
        backend = self._backend if self._backend in {"auto", "whisper_cpp", "faster_whisper"} else "auto"
        if backend in {"auto", "whisper_cpp"} and self._whisper_cpp_ready():
            try:
                return self._transcribe_with_whisper_cpp(audio_path, transcript_mode=transcript_mode)
            except Exception as exc:
                errors.append(f"whisper.cpp:{type(exc).__name__}")
                if backend == "whisper_cpp":
                    raise

        try:
            return self._transcribe_with_faster_whisper(audio_path, transcript_mode=transcript_mode)
        except Exception as exc:
            errors.append(f"faster_whisper:{type(exc).__name__}")
            if backend == "faster_whisper":
                raise
            raise RuntimeError("; ".join(errors))

    def _maybe_escalate_transcription(self, *, fast_text: str,
                                       audio_path: "Path",
                                       audio_seconds: float,
                                       transcript_mode: str) -> str:
        """Phase-3 wiring: after the fast tier produces a transcript,
        ask the TranscriptionRouter whether the content warrants a
        re-decode with the accurate tier (faster_whisper medium.en).

        Escalation reasons (per the router): clip > 6s, contains
        email/URL/path/digit-run, starts with a destructive verb,
        verbatim mode forced. Cheap when not needed (zero extra work).
        Best-effort: any failure returns the fast text unchanged.
        """
        try:
            from hgr.live_api.transcription_router import (
                TranscriptionRouter, TranscriptTier)
        except Exception:
            return fast_text
        try:
            router = TranscriptionRouter()
            decision = router.decide(
                audio_seconds=audio_seconds, fast_text=fast_text)
            if decision.tier == TranscriptTier.FAST:
                return fast_text
            # ACCURATE was chosen — re-decode with faster_whisper.
            # If THAT path also fails, return the original fast text
            # rather than the error.
            try:
                accurate = self._transcribe_with_faster_whisper(
                    audio_path, transcript_mode=transcript_mode)
            except Exception:
                return fast_text
            if not accurate or not accurate.strip():
                return fast_text
            # Log the escalation so we can observe its hit-rate.
            try:
                if hasattr(self, "_logger") and self._logger:
                    self._logger.event(
                        "voice_transcription_escalated",
                        reason=decision.reason,
                        fast_len=len(fast_text),
                        accurate_len=len(accurate))
            except Exception:
                pass
            return accurate
        except Exception:
            return fast_text

    def _transcribe_with_faster_whisper(self, audio_path: Path, *, transcript_mode: str = "command") -> str:
        model = self._ensure_model()
        # Adaptive decoding params. Earlier version used beam=1 for
        # short commands, which was too greedy: whisper produced typos
        # ("poekr" instead of "poker") AND duplicated-phrase
        # hallucinations ("play poekr face, play poker face") on
        # near-ambiguous audio. beam=3 fixes both with negligible
        # speed cost vs beam=1 on CUDA (~30-60ms extra).
        duration = _audio_file_duration_sec(audio_path)
        is_command = transcript_mode == "command"
        if is_command and duration > 0:
            beam = 3
            best_of = 3
        elif is_command:
            beam = 3
            best_of = 3
        else:
            beam = 5
            best_of = 5
        # Temperature ladder: (0.0, 0.2, 0.4) for BOTH modes. Greedy-
        # only for commands looked tempting (shorter utterances need
        # less retry) but in practice it just made borderline
        # commands return empty instead of recovering at temp=0.2.
        # OpenAI's reference Whisper uses (0.0, 0.2, 0.4, 0.6, 0.8,
        # 1.0); we keep just the first three for latency.
        temp_ladder = (0.0, 0.2, 0.4)
        # Wall-clock timing around the Whisper call. Adversarial review
        # of the VAD-timing tuning flagged that the temperature ladder
        # silently 2-3x's wall-clock on borderline audio (when temp=0.0
        # fails log_prob_threshold and falls through to 0.2 and 0.4).
        # Logging the actual inference time per command gives us
        # ground-truth data for future tuning, AND surfaces ladder-
        # escalation events as anomalously high values in logs.
        import time as _t
        _whisper_t0 = _t.monotonic()
        segments, _info = model.transcribe(
            str(audio_path),
            language="en",
            initial_prompt=self._build_initial_prompt(transcript_mode=transcript_mode),
            beam_size=beam,
            best_of=best_of,
            patience=1.0,
            temperature=temp_ladder,
            # Disable cross-segment conditioning so a previous turn's
            # garbage doesn't get re-emitted into this one. This is the
            # number-one cause of "play poker face" repeating itself.
            condition_on_previous_text=False,
            # Inner Silero VAD: OFF. We tried True and it correlated
            # with Whisper hallucinating "Please." and "I think so."
            # on real "play spotify" speech — the failure mode is
            # Silero rejecting the actual phonemes as non-speech,
            # leaving Whisper only the trailing silence/breath which
            # it fills with training-set common phrases. The outer
            # VAD in _record_to_wav already trims to actual speech
            # boundaries; Whisper handles internal silence fine.
            vad_filter=False,
            # Confidence-based suppression of hallucinations:
            #   - no_speech_threshold: skip segments whisper itself
            #     thinks are silence (default 0.6, explicit).
            #   - log_prob_threshold: drop segments below this
            #     average log-probability per token. Hallucinated
            #     repeats usually have low confidence.
            #   - compression_ratio_threshold: drop segments where
            #     output is suspiciously redundant (e.g. same word
            #     repeated). 2.4 is the OpenAI default; 2.2 here
            #     because command transcripts are short and a real
            #     command rarely compresses heavily.
            no_speech_threshold=0.6,
            log_prob_threshold=-1.0,
            compression_ratio_threshold=2.2,
        )
        # IMPORTANT: faster-whisper returns a GENERATOR. The actual
        # decode work happens when you iterate it. So we need to
        # materialize before stopping the timer below — otherwise the
        # timing would just record the generator-construction cost
        # (microseconds) and miss the real inference.
        segments = list(segments)
        try:
            _whisper_ms = int((_t.monotonic() - _whisper_t0) * 1000)
            import sys as _sys
            _sys.stderr.write(f"[voice] whisper_inference_ms={_whisper_ms}\n")
            _sys.stderr.flush()
        except Exception:
            pass
        # All non-empty segments pass through. The earlier per-segment
        # avg_logprob<-1.0 filter was hiding what Whisper actually
        # transcribed — user got "command not understood" with no
        # diagnostic trail. The existing in-decoder log_prob_threshold
        # already rejects whole passes that are too low-confidence;
        # adding a client-side per-segment gate on top was redundant
        # at best and silenced legitimate marginal transcripts at
        # worst. Log the per-segment confidence so we can see what
        # Whisper thought of the input.
        parts: list[str] = []
        for segment in segments:
            text_piece = (segment.text or "").strip()
            if not text_piece:
                continue
            try:
                avg_lp = float(getattr(segment, "avg_logprob", 0.0) or 0.0)
                no_speech = float(getattr(segment, "no_speech_prob", 0.0) or 0.0)
                import sys as _sys
                _sys.stderr.write(
                    f"[voice] segment avg_logprob={avg_lp:.2f} "
                    f"no_speech_prob={no_speech:.2f} text={text_piece!r}\n"
                )
                _sys.stderr.flush()
            except Exception:
                pass
            parts.append(text_piece)
        text = self._normalize_text(" ".join(parts), transcript_mode=transcript_mode)
        # Last-resort dedup pass: if whisper still emitted a duplicated
        # phrase ("play poker face, play poker face"), collapse to
        # one. Cheap; runs only when commas / near-duplicates present.
        if is_command:
            text = _dedup_repeated_phrase(text)
        return text

    def _ensure_model(self):
        if self._model is not None:
            return self._model

        from faster_whisper import WhisperModel

        self._model_root.mkdir(parents=True, exist_ok=True)
        # Pick the best available device. CUDA cuts transcription time
        # 2-3x on supported NVIDIA cards (medium model: 1.5s CPU vs
        # 0.5s GPU). faster-whisper falls back to CPU automatically
        # when CUDA libs are missing, so trying CUDA first is safe.
        device, compute_type = _pick_whisper_device()
        errors: list[str] = []
        for candidate in self._model_candidates:
            for dev, ctype in ((device, compute_type), ("cpu", "int8")):
                try:
                    self._message = (f"loading whisper model {candidate}"
                                     f" [{dev}/{ctype}]")
                    self._model = WhisperModel(
                        candidate,
                        device=dev,
                        compute_type=ctype,
                        download_root=str(self._model_root),
                        # macOS: cap CTranslate2 to 4 CPU threads so a voice
                        # transcription doesn't saturate every core and starve
                        # the gesture MediaPipe pipeline (both are CPU-bound on
                        # Apple Silicon), which was tanking fps whenever a voice
                        # command ran. 0 = faster-whisper default (all cores) on
                        # every other platform, so Windows behavior is unchanged.
                        cpu_threads=(4 if platform.system() == "Darwin" else 0),
                    )
                    self._model_name = candidate
                    return self._model
                except Exception as exc:
                    errors.append(f"{candidate}@{dev}:{type(exc).__name__}")
                    # No point retrying with CPU if we already are.
                    if dev == "cpu":
                        break
        raise RuntimeError("no whisper model could be loaded: " + ", ".join(errors))

    def _whisper_cpp_ready(self) -> bool:
        return self._resolve_whisper_cpp_command() is not None and self._resolve_whisper_cpp_model_path() is not None

    def _candidate_whisper_roots(self) -> list[Path]:
        roots: list[Path] = []
        # PyInstaller bundle location — the installed app puts whisper.cpp
        # under the _internal folder (see builder/windows/hgr_app.spec).
        base = app_base_path()
        roots.append(base)
        roots.append(base / "whisper.cpp")
        # Source-checkout fallback: walk up from this file to let `python
        # run_app.py` keep finding `<repo>/whisper.cpp/`.
        here = Path(__file__).resolve()
        for parent in here.parents:
            if parent not in roots:
                roots.append(parent)
                roots.append(parent / "whisper.cpp")
        # Last, the legacy dev-machine path used before bundle-awareness.
        home_candidate = Path.home() / "Documents" / "whisper.cpp"
        if home_candidate not in roots:
            roots.append(home_candidate)
        return roots

    def _resolve_whisper_cpp_command(self) -> tuple[str, ...] | None:
        if self._whisper_cpp_command:
            return self._whisper_cpp_command
        env_value = str(os.getenv("HGR_WHISPER_CPP", "") or "").strip()
        if env_value:
            path = Path(env_value)
            if path.exists():
                return (str(path),)
            resolved = shutil.which(env_value)
            if resolved:
                return (resolved,)
        for command_name in ("whisper-cli.exe", "whisper-cli"):
            resolved = shutil.which(command_name)
            if resolved:
                return (resolved,)
        for root in self._candidate_whisper_roots():
            for build_dir in ("build", "build_cuda", "build_vulkan", "build_stream"):
                bin_dir = root / build_dir / "bin"
                for candidate in (bin_dir / "Release" / "whisper-cli.exe", bin_dir / "whisper-cli.exe"):
                    if candidate.exists():
                        return (str(candidate),)
        return None

    def _resolve_whisper_cpp_model_path(self) -> Path | None:
        if self._whisper_cpp_model_path is not None and self._whisper_cpp_model_path.exists():
            return self._whisper_cpp_model_path
        env_value = str(os.getenv("HGR_WHISPER_CPP_MODEL", "") or "").strip()
        if env_value:
            path = Path(env_value)
            if path.exists():
                self._whisper_cpp_model_path = path
                return path
        candidate_roots: list[Path] = [self._model_root]
        for root in self._candidate_whisper_roots():
            candidate_roots.append(root / "models")
        for root in candidate_roots:
            for candidate_name in (
                "ggml-medium.en.bin",
                "ggml-small.en.bin",
                "ggml-base.en.bin",
                "ggml-medium.bin",
                "ggml-small.bin",
                "ggml-base.bin",
            ):
                candidate = root / candidate_name
                if candidate.exists():
                    self._whisper_cpp_model_path = candidate
                    return candidate
            extras = sorted(path for path in root.glob("ggml-*.bin") if not path.name.startswith("for-tests-"))
            if extras:
                self._whisper_cpp_model_path = extras[0]
                return extras[0]
        return None

    def _transcribe_with_whisper_cpp(self, audio_path: Path, *, transcript_mode: str = "command") -> str:
        command = self._resolve_whisper_cpp_command()
        model_path = self._resolve_whisper_cpp_model_path()
        if command is None or model_path is None:
            raise RuntimeError("whisper.cpp backend not ready")
        vad_model_path = self._resolve_whisper_cpp_vad_model_path()
        self._message = f"running whisper.cpp ({model_path.name})"
        thread_count = max(2, min(8, (os.cpu_count() or 4)))
        prompt = self._build_initial_prompt(transcript_mode=transcript_mode)[:500]
        # Match the faster-whisper command-mode beam/best-of values
        # (3/3 vs the prior 6/6). The original 6/6 was a different
        # quality target than the faster-whisper backend, producing
        # silently-different transcripts when the two backends were
        # both active on the same machine. 3/3 matches what
        # faster-whisper does for short command-mode phrases and
        # cuts whisper.cpp inference time roughly 30-40% for the
        # same accuracy on <5-word commands. Dictation mode still
        # uses 6/6 (caller passes transcript_mode="dictation").
        is_command_mode = transcript_mode == "command"
        bs = "3" if is_command_mode else "6"
        bo = "3" if is_command_mode else "6"
        whisper_command = [
            *command,
            "-m",
            str(model_path),
            "-f",
            str(audio_path),
            "-l",
            "en",
            "-t",
            str(thread_count),
            "-bs",
            bs,
            "-bo",
            bo,
            "-mc",
            "128",
            "-ml",
            "96",
            "-sow",
            "-nf",
            "-sns",
            "-nt",
            "-np",
            "--prompt",
            prompt,
        ]
        if vad_model_path is not None:
            whisper_command.extend(
                [
                    "--vad",
                    "-vm",
                    str(vad_model_path),
                    "-vt",
                    "0.55",
                    "-vspd",
                    "180",
                    "-vsd",
                    "160",
                    "-vp",
                    "50",
                ]
            )
        completed = subprocess.run(
            whisper_command,
            capture_output=True,
            text=True,
            timeout=45.0,
            check=False,
            **hidden_subprocess_kwargs(),
        )
        if completed.returncode != 0 and not completed.stdout.strip():
            raise RuntimeError("whisper.cpp transcription failed")
        lines = []
        for line in completed.stdout.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            match = re.match(r"^\[[0-9:.]+\s+-->\s+[0-9:.]+\]\s*(.*)$", stripped)
            if match is not None:
                fragment = match.group(1).strip()
                if fragment:
                    lines.append(fragment)
        if not lines:
            for line in completed.stdout.splitlines():
                stripped = line.strip()
                if stripped and not stripped.startswith(("whisper_", "system_info:", "main:", "encode_", "decode_")):
                    lines.append(stripped)
        return self._normalize_text(" ".join(lines), transcript_mode=transcript_mode)

    def _resolve_whisper_cpp_vad_model_path(self) -> Path | None:
        if self._whisper_cpp_vad_model_path is not None and self._whisper_cpp_vad_model_path.exists():
            return self._whisper_cpp_vad_model_path
        env_value = str(os.getenv("HGR_WHISPER_CPP_VAD_MODEL", "") or "").strip()
        if env_value:
            path = Path(env_value)
            if path.exists():
                self._whisper_cpp_vad_model_path = path
                return path
        for root in (self._model_root, self._whisper_cpp_root / "models"):
            for candidate_name in (
                "ggml-silero-v5.1.2.bin",
                "ggml-silero-v6.2.0.bin",
            ):
                candidate = root / candidate_name
                if candidate.exists():
                    self._whisper_cpp_vad_model_path = candidate
                    return candidate
        return None

    def _build_initial_prompt(self, *, transcript_mode: str = "command") -> str:
        if transcript_mode == "dictation":
            return (
                "Transcribe natural spoken dictation for emails, essays, messages, and speeches. "
                "Dictation may include spoken punctuation like comma, period, question mark, "
                "new line, and new paragraph."
            )
        if transcript_mode == "save_prompt":
            return (
                "Transcribe a save-location reply. "
                "The speaker may say auto, default, cancel, delete, nevermind, "
                "or a folder name such as desktop, documents, downloads, pictures, videos, "
                "onedrive, or an absolute Windows path."
            )
        if transcript_mode == "playlist":
            return (
                "Transcribe only the spoken Spotify playlist title. "
                "Return just the playlist name with no extra words like play, add, remove, current, and, or playlist."
            )
        # Concrete example sentences (not just a vocabulary list)
        # so Whisper biases toward outputting the leading verb. Bare
        # vocab lists let Whisper drop short low-prosody first words
        # like "play" / "open" / "close" — the model reads the
        # opening syllable as breath and trims it. Worked examples
        # show the canonical sentence shape and make the verb token
        # near-certain.
        base = (
            "Short spoken commands always start with a verb. "
            "Examples: 'play Poker Face on Spotify.' "
            "'open Chrome.' 'open YouTube.' 'close Discord.' "
            "'search for cat videos on YouTube.' 'go to gmail.' "
            "'increase volume to fifty.' 'mute volume.' "
            "'clip that.' 'clip the last minute.' 'save clip.' "
            "'save the last 2 minute clip.' 'clip the past 5 minutes.' "
            "'clip the last 30 seconds.' "
            "Verbs include open, launch, start, run, boot up, fire up, "
            "load, pull up, show me, bring up, switch to, focus on, "
            "close, exit, quit, play, put on, listen to, queue, "
            "search, search for, search up, look up, find, go to, "
            "navigate to, clip, clip that, clip the last minute, "
            "save clip, make a clip, increase, decrease, set, mute, "
            "unmute. "
            "Targets include Spotify, Chrome, Edge, Firefox, Settings, "
            "File Explorer, Outlook, Discord, Steam, Notepad, Word, "
            "Excel, PowerPoint, Visual Studio, Visual Studio Code, "
            "GitHub, Gmail, YouTube, ChatGPT, Reddit, Touchless, and "
            "file or folder names like Desktop, Documents, Downloads, "
            "Pictures, Videos, resume, budget, invoice, project notes, "
            "homework."
        )
        if not self._app_hints:
            return base
        hint_text = ", ".join(self._app_hints[:24])
        return f"{base} Installed apps include {hint_text}."

    def _estimate_noise_floor(self, ambient_levels: list[float]) -> float:
        if not ambient_levels:
            return 0.003
        window = ambient_levels[-6:]
        return max(0.0025, float(np.median(np.asarray(window, dtype=np.float32))))

    def _normalize_text(self, text: str, *, transcript_mode: str = "command") -> str:
        normalized = " ".join(word for word in str(text or "").replace("\n", " ").split() if word).strip()
        if transcript_mode == "dictation":
            return normalized
        if transcript_mode == "playlist":
            lowered = f" {normalized.lower()} "
            replacements = (
                ("feel good", "feel-good"),
                ("r and b", "r&b"),
                ("hip hop", "hip-hop"),
            )
            for source, target in replacements:
                lowered = lowered.replace(f" {source} ", f" {target} ")
            lowered = re.sub(r"^[^a-z0-9]+", " ", lowered)
            lowered = re.sub(r"(and|then|uh|um|please|called|named|titled|playlist|spotify|current|the|my|to|from|add|remove|it|this)", " ", lowered)
            lowered = re.sub(r"\s+", " ", lowered).strip(" .!?-_")
            return " ".join(lowered.split()).strip()
        lowered = f" {normalized.lower()} "
        replacements = (
            ("google chrome", "chrome"),
            ("file explore", "file explorer"),
            ("files explorer", "file explorer"),
            ("blu tooth", "bluetooth"),
            ("wi fi", "wifi"),
            ("e mail", "email"),
            ("a c dc", "ac/dc"),
            ("ac dc", "ac/dc"),
            ("key card", "kicad"),
            ("key cards", "kicad"),
            ("key cad", "kicad"),
            ("ki cad", "kicad"),
            ("k i cad", "kicad"),
            ("clothes", "close"),
            ("cloths", "close"),
            ("poll up", "pull up"),
            ("pool up", "pull up"),
            ("full up", "pull up"),
            ("pulled up", "pull up"),
            ("booted up", "boot up"),
            ("boot it up", "boot up"),
            ("boots up", "boot up"),
            ("fired up", "fire up"),
            ("fires up", "fire up"),
            ("load up", "launch"),
            ("loaded up", "launch"),
            ("load it up", "launch"),
            ("show me the", "show me"),
            ("show-me", "show me"),
            ("showed me", "show me"),
            ("bring up the", "bring up"),
            ("bring it up", "bring up"),
            ("brought up", "bring up"),
            ("switch over to", "switch to"),
            ("switching to", "switch to"),
            ("jump to", "switch to"),
            ("hop onto", "switch to"),
            ("give me", "show me"),
            ("get me", "open"),
            ("go on to", "go to"),
            ("take me over to", "go to"),
            ("search for the", "search for"),
            ("look me up", "look up"),
            ("lookup", "look up"),
            ("searchup", "search up"),
            ("put on some", "put on"),
        )
        for source, target in replacements:
            lowered = lowered.replace(f" {source} ", f" {target} ")
        lowered = re.sub(r"\b(show)(me)(?=\S)", r"\1 \2 ", lowered)
        lowered = re.sub(r"\b(open|close|launch|start|boot|pull|show|fire|bring|run|load)(?=[a-z0-9])", lambda m: m.group(1) + " ", lowered)
        lowered = re.sub(r"\bshowme\b", "show me", lowered)
        lowered = re.sub(r"\bpullup\b", "pull up", lowered)
        lowered = re.sub(r"\bbootup\b", "boot up", lowered)
        lowered = re.sub(r"\bfireup\b", "fire up", lowered)
        lowered = re.sub(r"\bbringup\b", "bring up", lowered)
        lowered = re.sub(r"\s+", " ", lowered)
        return " ".join(lowered.split()).strip()

    def _normalize_hint_name(self, text: str) -> str:
        value = str(text or "").strip()
        if not value:
            return ""
        value = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", value)
        value = re.sub(r"(?<=[A-Z])(?=[A-Z][a-z])", " ", value)
        value = re.sub(r"[^A-Za-z0-9+/.-]+", " ", value)
        value = " ".join(value.split()).strip().lower()
        replacements = (
            ("ki cad", "kicad"),
            ("key cad", "kicad"),
            ("visual studios", "visual studio"),
            ("chat gpt", "chatgpt"),
            ("clothes", "close"),
            ("cloths", "close"),
        )
        for source, target in replacements:
            value = value.replace(source, target)
        return value

# Author: Konstantin Markov
