from __future__ import annotations

import json
import os
import platform
import re
import shutil
import subprocess
import tempfile
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
    """
    try:
        import sounddevice as sd
    except Exception:
        return []

    try:
        devices = sd.query_devices()
    except Exception:
        return []

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
    return names


class VoiceCommandListener:
    def __init__(
        self,
        *,
        backend: str = "auto",
        model_name: str = "distil-large-v3",
        model_fallbacks: tuple[str, ...] = ("small.en",),
        sample_rate: int = 48000,
        block_duration: float = 0.08,
        min_voice_seconds: float = 0.32,
        min_command_seconds: float = 0.68,
        end_silence_seconds: float = 3.0,
        start_timeout_seconds: float = 5.0,
        whisper_cpp_command: tuple[str, ...] | None = None,
        whisper_cpp_model_path: Path | None = None,
        preferred_input_device: str | int | None = None,
        input_gain: float = 1.0,
    ) -> None:
        self._available = platform.system() == "Windows"
        self._message = "voice idle"
        self._backend = str(backend or "auto").strip().lower()
        self._model_name = model_name
        self._model_candidates = tuple(dict.fromkeys((model_name, *model_fallbacks)))
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
        if isinstance(preferred_input_device, str) and preferred_input_device.strip():
            self.set_input_device_name(preferred_input_device)
        elif isinstance(preferred_input_device, int):
            self._preferred_input_device_index = preferred_input_device
        # Optional override: when set, the voice pipeline reads PCM from
        # this source instead of sounddevice. Used for the phone-camera
        # QR flow's mic stream.
        self._external_audio_source = None

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

    def input_device_name(self) -> str | None:
        return self._preferred_input_device_name

    def set_input_gain(self, gain: float) -> None:
        try:
            value = float(gain)
        except (TypeError, ValueError):
            value = 1.0
        self._input_gain = max(0.1, min(10.0, value))

    @property
    def input_gain(self) -> float:
        return self._input_gain

    def _resolve_input_device_index(self, device_name: str | None) -> int | None:
        if not device_name:
            return None
        try:
            import sounddevice as sd
        except Exception:
            return None
        try:
            devices = sd.query_devices()
        except Exception:
            return None
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
            self._ensure_model()
            self._message = f"voice ready: {self._model_name}"
        except Exception:
            pass

    def listen(
        self,
        *,
        max_seconds: float = 12.0,
        status_callback: Callable[[str], None] | None = None,
        stop_event=None,
        transcript_mode: str = "command",
    ) -> VoiceCommandResult:
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
        except Exception:
            transcription_failed = True
            fallback = self._fallback_system_speech(max_seconds=max_seconds, transcript_mode=transcript_mode)
            if fallback.success:
                try:
                    audio_path.unlink(missing_ok=True)
                except Exception:
                    pass
                return fallback
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
            fallback = self._fallback_system_speech(max_seconds=max_seconds, transcript_mode=transcript_mode)
            if fallback.success:
                return fallback
            self._message = "dictation not understood" if transcript_mode == "dictation" else "voice command not understood"
            return VoiceCommandResult(heard_text="", success=False, message=self._message)

        self._message = f"heard: {text}"
        return VoiceCommandResult(heard_text=text, success=True, message=self._message)

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
        ambient_levels: list[float] = []
        chunks: list[np.ndarray] = []
        preroll_blocks = max(4, int(0.60 / self._block_duration))
        preroll: deque[np.ndarray] = deque(maxlen=preroll_blocks)
        max_rms_seen = 0.0
        final_noise_floor = 0.0
        final_trigger_threshold = 0.0

        stream_kwargs = {
            "samplerate": sample_rate,
            "channels": 1,
            "dtype": "float32",
            "blocksize": block_size,
        }
        if self._preferred_input_device_index is not None:
            stream_kwargs["device"] = self._preferred_input_device_index

        # Route through the phone-posted PCM source instead of
        # sounddevice when one has been installed. The external source
        # exposes the same read(frames) -> (ndarray (frames,1) float32,
        # overflow) contract that sd.InputStream.read returns, so no
        # changes to the loop below are needed.
        external = self._external_audio_source
        if external is not None and hasattr(external, "drain"):
            # Discard any audio buffered before this session started —
            # otherwise a tap sound or idle chatter from seconds ago
            # fires voice activation on the very first read and the
            # loop enters the recording path with no real speech.
            external.drain()
        stream_ctx = external if external is not None else sd.InputStream(**stream_kwargs)

        with stream_ctx as stream:
            for block_index in range(max_blocks):
                if stop_event is not None and stop_event.is_set():
                    if voice_started and chunks:
                        break
                    return None
                data, _overflow = stream.read(block_size)
                mono = np.squeeze(np.asarray(data, dtype=np.float32))
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

                noise_floor = self._estimate_noise_floor(ambient_levels)
                trigger_threshold = max(noise_floor * 2.0, 0.008)
                silence_threshold = max(noise_floor * 1.3, 0.005)
                final_noise_floor = noise_floor
                final_trigger_threshold = trigger_threshold
                if rms > max_rms_seen:
                    max_rms_seen = rms

                if voice_started:
                    chunks.append(mono.copy())
                    voice_blocks += 1
                    if rms <= silence_threshold:
                        silence_blocks += 1
                    else:
                        silence_blocks = 0
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
                    if external is None and transcript_mode in {"command", "save_prompt"}:
                        required_silence = min(required_silence, 2.0)
                    if (
                        active_seconds >= min_active_seconds
                        and silence_blocks * self._block_duration >= required_silence
                    ):
                        break
                elif rms >= trigger_threshold:
                    voice_started = True
                    if preroll:
                        chunks.extend(preroll)
                        voice_blocks = len(preroll)
                        preroll.clear()
                    else:
                        voice_blocks = 0
                    chunks.append(mono.copy())
                    voice_blocks += 1
                    silence_blocks = 0
                elif block_index + 1 >= start_timeout_blocks:
                    break
                else:
                    preroll.append(mono.copy())

        import sys as _sys
        source_desc = (
            "phone-external"
            if external is not None
            else f"local-sd(idx={self._preferred_input_device_index} name={self._preferred_input_device_name!r})"
        )
        print(
            f"[voice] mode={transcript_mode} source={source_desc} gain={self._input_gain:.2f} "
            f"noise_floor={final_noise_floor:.4f} trigger={final_trigger_threshold:.4f} "
            f"max_rms_seen={max_rms_seen:.4f} voice_started={voice_started} chunks={len(chunks)}",
            file=_sys.stderr,
            flush=True,
        )

        if not voice_started or not chunks:
            return None

        audio = np.concatenate(chunks).astype(np.float32, copy=False)
        peak = float(np.max(np.abs(audio))) if audio.size else 0.0
        if peak <= 0.0025:
            return None
        # Detect input clipping (max RMS seen > 0.9 strongly suggests
        # the device is clipping mid-capture, which whisper turns
        # into garbled or empty transcripts — a common failure mode
        # for headset mics with their own preamp set too high).
        # Log a one-line warning so the user / dev can see it in the
        # detailed log without changing app behaviour.
        if max_rms_seen >= 0.85 or peak >= 0.995:
            import sys as _sys
            print(
                f"[voice] WARNING input likely clipping (max_rms={max_rms_seen:.3f} "
                f"peak={peak:.3f}). Headset mics with high gain can produce distorted "
                f"audio that whisper can't transcribe; lower the device's input volume "
                f"in Windows Sound Settings OR drop Touchless 'Mic input gain' below 1.0.",
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
        target_peak = 0.95
        if peak > 0.0:
            audio = audio * (target_peak / peak)

        with tempfile.NamedTemporaryFile(prefix="hgr_voice_", suffix=".wav", delete=False) as tmp:
            path = Path(tmp.name)
        with wave.open(str(path), "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_data = np.clip(audio * 32767.0, -32768.0, 32767.0).astype(np.int16)
            wav_file.writeframes(wav_data.tobytes())
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

    def _transcribe_with_faster_whisper(self, audio_path: Path, *, transcript_mode: str = "command") -> str:
        model = self._ensure_model()
        segments, _info = model.transcribe(
            str(audio_path),
            language="en",
            initial_prompt=self._build_initial_prompt(transcript_mode=transcript_mode),
            beam_size=5,
            best_of=5,
            patience=1.0,
            temperature=0.0,
            condition_on_previous_text=False,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 250},
        )
        parts = [segment.text.strip() for segment in segments if segment.text and segment.text.strip()]
        return self._normalize_text(" ".join(parts), transcript_mode=transcript_mode)

    def _ensure_model(self):
        if self._model is not None:
            return self._model

        from faster_whisper import WhisperModel

        self._model_root.mkdir(parents=True, exist_ok=True)
        errors: list[str] = []
        for candidate in self._model_candidates:
            try:
                self._message = f"loading whisper model {candidate}"
                self._model = WhisperModel(
                    candidate,
                    device="cpu",
                    compute_type="int8",
                    download_root=str(self._model_root),
                )
                self._model_name = candidate
                return self._model
            except Exception as exc:
                errors.append(f"{candidate}:{type(exc).__name__}")
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
            "6",
            "-bo",
            "6",
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
        base = (
            "Short spoken commands. Verbs include open, launch, start, run, boot up, "
            "fire up, load, pull up, show me, bring up, switch to, focus on, close, "
            "exit, quit, play, put on, listen to, queue, search, search for, search up, "
            "look up, find, go to, navigate to, clip, clip that, clip the last minute, "
            "save clip, make a clip. "
            "Targets include Spotify, Chrome, Edge, Firefox, Settings, File Explorer, "
            "Outlook, Discord, Steam, Notepad, Word, Excel, PowerPoint, Visual Studio, "
            "Visual Studio Code, GitHub, Gmail, YouTube, ChatGPT, Reddit, and file or "
            "folder names like Desktop, Documents, Downloads, Pictures, Videos, resume, "
            "budget, invoice, project notes, homework."
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

    def _fallback_system_speech(self, *, max_seconds: float, transcript_mode: str = "command") -> VoiceCommandResult:
        import base64
        import subprocess

        command = [
            "powershell.exe",
            "-NoProfile",
            "-NonInteractive",
            "-ExecutionPolicy",
            "Bypass",
            "-EncodedCommand",
            self._encoded_system_speech_script(max_seconds=max_seconds),
        ]
        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=max_seconds + 4.0,
                check=False,
                **hidden_subprocess_kwargs(),
            )
        except Exception:
            return VoiceCommandResult(heard_text="", success=False, message="voice command not heard")

        payload = self._parse_payload(completed.stdout)
        if not payload or payload.get("error"):
            return VoiceCommandResult(heard_text="", success=False, message="voice command not heard")
        heard_text = self._select_phrase(payload.get("phrases") or [], transcript_mode=transcript_mode)
        if not heard_text:
            return VoiceCommandResult(heard_text="", success=False, message="voice command not heard")
        return VoiceCommandResult(heard_text=heard_text, success=True, message=f"heard: {heard_text}")

    def _parse_payload(self, stdout_text: str) -> dict | None:
        lines = [line.strip() for line in (stdout_text or "").splitlines() if line.strip()]
        for line in reversed(lines):
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                return value
        return None

    def _select_phrase(self, phrases: list[dict], *, transcript_mode: str = "command") -> str:
        best_text = ""
        best_score = -1.0
        structural_phrases = (
            "file",
            "folder",
            "documents",
            "downloads",
            "desktop",
            "outlook",
            "sent items",
            "inbox",
            "settings",
        )
        for item in phrases:
            if not isinstance(item, dict):
                continue
            text = self._normalize_text(str(item.get("text", "")), transcript_mode=transcript_mode)
            if not text:
                continue
            confidence = float(item.get("confidence", 0.0) or 0.0)
            word_count = len(text.split())
            hint_bonus = min(0.16, sum(0.04 for hint in self._app_hints[:20] if hint and hint in text))
            structure_bonus = min(0.12, sum(0.03 for phrase in structural_phrases if phrase in text))
            chain_bonus = 0.05 if len(re.findall(r"\b(?:in|inside|under|within)\b", text)) >= 2 else 0.0
            score = (
                confidence
                + min(word_count, 16) * 0.075
                + min(len(text), 120) * 0.0015
                + hint_bonus
                + structure_bonus
                + chain_bonus
            )
            if score > best_score:
                best_score = score
                best_text = text
        return best_text

    def _encoded_system_speech_script(self, *, max_seconds: float) -> str:
        import base64

        seconds = max(6.0, float(max_seconds))
        script = f"""
$ErrorActionPreference = 'Stop'
try {{
    Add-Type -AssemblyName System.Speech
    $culture = [System.Globalization.CultureInfo]::GetCultureInfo('en-US')
    $recognizer = New-Object System.Speech.Recognition.SpeechRecognitionEngine($culture)
    $grammar = New-Object System.Speech.Recognition.DictationGrammar
    $recognizer.LoadGrammar($grammar)
    $recognizer.SetInputToDefaultAudioDevice()
    $recognizer.InitialSilenceTimeout = [TimeSpan]::FromSeconds(3.0)
    $recognizer.BabbleTimeout = [TimeSpan]::FromSeconds(3.0)
    $recognizer.EndSilenceTimeout = [TimeSpan]::FromSeconds(1.20)
    $recognizer.EndSilenceTimeoutAmbiguous = [TimeSpan]::FromSeconds(1.55)
    $deadline = [DateTime]::UtcNow.AddSeconds({seconds})
    $phrases = New-Object System.Collections.Generic.List[object]
    while ([DateTime]::UtcNow -lt $deadline) {{
        $remaining = $deadline - [DateTime]::UtcNow
        if ($remaining.TotalSeconds -lt 1) {{ break }}
        try {{
            $result = $recognizer.Recognize([TimeSpan]::FromSeconds([Math]::Min(4.5, $remaining.TotalSeconds)))
        }} catch {{
            $result = $null
        }}
        if ($null -ne $result -and -not [string]::IsNullOrWhiteSpace($result.Text)) {{
            $phrases.Add([PSCustomObject]@{{
                text = $result.Text
                confidence = [double]$result.Confidence
            }})
            $wordCount = $result.Text.Trim().Split().Count
            $confidence = [double]$result.Confidence
            if (
                ($wordCount -ge 8 -and $confidence -ge 0.45) -or
                ($wordCount -ge 6 -and $confidence -ge 0.62)
            ) {{
                break
            }}
        }}
    }}
    $recognizer.Dispose()
    [PSCustomObject]@{{
        phrases = $phrases
        error = $null
    }} | ConvertTo-Json -Compress -Depth 4
}} catch {{
    [PSCustomObject]@{{
        phrases = @()
        error = $_.Exception.Message
    }} | ConvertTo-Json -Compress -Depth 4
}}
"""
        return base64.b64encode(script.encode("utf-16-le")).decode("ascii")

# Author: Konstantin Markov
