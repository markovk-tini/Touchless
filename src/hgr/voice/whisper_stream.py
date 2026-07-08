from __future__ import annotations

import os
import queue
import re
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Tuple

import numpy as np
import sounddevice as sd

from ..utils.runtime_paths import app_base_path


@dataclass(frozen=True)
class DictationEvent:
    event: str  # ready | hypothesis | final | error | stopped
    text: str = ""
    confidence: float = 0.0


_SAMPLE_RATE = 16000
_BLOCK_MS = 100
_BLOCK_SAMPLES = _SAMPLE_RATE * _BLOCK_MS // 1000
_SILENCE_COMMIT_MS = 1000
_MAX_UTTERANCE_MS = 30000
_MIN_SPEECH_MS = 500
_RMS_SILENCE_THRESHOLD = 0.003
_MODEL_ID = "deepdml/faster-whisper-large-v3-turbo-ct2"

# Kill switch for live dictation hypotheses (interim results). Consumed by
# app.integration.noop_engine (imported as _DICTATION_HYP_ENABLED). Set
# HGR_DICTATION_HYPOTHESES=0 to disable. Defined here so the module stays
# self-consistent with its importers — the committed code (both main and this
# branch) imports this name but the full dictation-streaming revamp that also
# defines it currently lives only in the Windows working tree.
_HYP_ENABLED = os.getenv("HGR_DICTATION_HYPOTHESES", "1").strip() != "0"

# Hotwords bias the decoder toward listed terms. A tech-heavy
# default list used to live here, but it was net-negative for
# natural prose dictation: when the user said something acoustically
# close to a hotword, the bias pushed the wrong term into the output.
# Leave it empty by default; power users can opt in with
# HGR_WHISPER_HOTWORDS="Qwen Llama vcpkg ..." for their own jargon.
_DEFAULT_HOTWORDS = ""


def _resolve_hotwords() -> Optional[str]:
    raw = os.getenv("HGR_WHISPER_HOTWORDS")
    if raw is None:
        raw = _DEFAULT_HOTWORDS
    raw = raw.strip()
    return raw or None


def _resolve_model_dir() -> Path:
    env_dir = os.getenv("HGR_WHISPER_MODEL_DIR", "").strip()
    if env_dir:
        return Path(env_dir)
    return Path.home() / "Documents" / "HGRVoiceModels"


def _match_sd_input_device(preferred_name: str) -> Optional[int]:
    if not preferred_name:
        return None
    target = preferred_name.lower().strip()
    try:
        devices = sd.query_devices()
    except Exception as exc:
        print(f"[whisper-stream] sd.query_devices failed: {exc}")
        return None
    inputs: List[Tuple[int, str]] = [
        (i, d["name"]) for i, d in enumerate(devices) if d.get("max_input_channels", 0) > 0
    ]
    for idx, name in inputs:
        if name.lower().strip() == target:
            return idx
    for idx, name in inputs:
        lowered = name.lower()
        if target in lowered or lowered in target:
            return idx
    tokens = [t for t in re.split(r"[^a-z0-9]+", target) if len(t) > 2]
    if tokens:
        for idx, name in inputs:
            lowered = name.lower()
            if all(tok in lowered for tok in tokens):
                return idx
    return None


class WhisperStreamer:
    """Local streaming dictation using faster-whisper.

    Commit-only decoding: one decode per utterance, triggered by RMS-based
    silence detection. Audio capture is in-process via sounddevice; decode
    is in-process via faster-whisper (CTranslate2).
    """

    def __init__(
        self,
        *,
        preferred_microphone_name: Optional[str] = None,
        **_ignored,
    ) -> None:
        self._preferred_mic_name = (preferred_microphone_name or "").strip() or None
        self._available = False
        self._message = "faster-whisper not available"
        self._backend: Optional[str] = None
        self._model = None
        self._mic_index: Optional[int] = None
        self._hotwords = _resolve_hotwords()
        self._model_lock = threading.Lock()

        try:
            import ctranslate2
            cuda_ok = ctranslate2.get_cuda_device_count() > 0
        except Exception as exc:
            self._message = f"ctranslate2 import failed: {exc}"
            return

        self._device = "cuda" if cuda_ok else "cpu"
        self._compute_type = "int8_float16" if cuda_ok else "int8"
        self._backend = "cuda" if cuda_ok else "cpu"

        if self._preferred_mic_name:
            try:
                self._mic_index = _match_sd_input_device(self._preferred_mic_name)
                if self._mic_index is not None:
                    print(f"[whisper-stream] mic routed to sounddevice idx {self._mic_index} for '{self._preferred_mic_name}'")
                else:
                    print(f"[whisper-stream] no mic match for '{self._preferred_mic_name}' — using default input")
            except Exception as exc:
                print(f"[whisper-stream] mic resolve failed: {exc}")

        self._available = True
        self._message = f"faster-whisper ready ({self._backend})"

    @property
    def available(self) -> bool:
        return self._available

    @property
    def message(self) -> str:
        return self._message

    @property
    def backend(self) -> Optional[str]:
        return self._backend

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        from faster_whisper import WhisperModel

        model_root = _resolve_model_dir()
        model_root.mkdir(parents=True, exist_ok=True)
        print(f"[whisper-stream] loading {_MODEL_ID} on {self._device}/{self._compute_type} (dir={model_root})")
        t0 = time.monotonic()
        self._model = WhisperModel(
            _MODEL_ID,
            device=self._device,
            compute_type=self._compute_type,
            download_root=str(model_root),
        )
        print(f"[whisper-stream] model loaded in {time.monotonic() - t0:.1f}s")

    def _transcribe(self, audio: np.ndarray) -> List[str]:
        assert self._model is not None
        # Accuracy levers tuned per device. Latency-rebalanced 2026-05-12
        # after the user reported 10-15 s output times with beam_size=5
        # + 2 s silence-commit; halving target was 5-7 s.
        #   * beam_size: 3 on GPU (keeps ~75 % of the WER win that 5
        #     gave over greedy, at ~60 % of the inference time),
        #     1 on CPU (greedy is the only thing CPU users can afford
        #     and still feel responsive; VAD pre-filter + hallucination
        #     thresholds carry the accuracy load instead).
        #   * vad_filter=True (kept): strips silence/ambient noise
        #     before the decoder. Biggest single accuracy win, AND
        #     speeds up decode because we don't transcribe dead air.
        #   * compression_ratio_threshold + log_prob_threshold (kept):
        #     decoder-level anti-hallucination gates.
        beam_size = 3 if self._device == "cuda" else 1
        with self._model_lock:
            segments, _info = self._model.transcribe(
                audio,
                language="en",
                beam_size=beam_size,
                temperature=0.0,
                condition_on_previous_text=False,
                vad_filter=True,
                vad_parameters={
                    # Keep brief mid-sentence pauses (~700 ms or less)
                    # inside a single segment so the model has the
                    # full clause's acoustic context. Without this,
                    # short pauses split a sentence into two segments
                    # and word boundaries on either side of the
                    # pause come out garbled.
                    "min_silence_duration_ms": 700,
                    # Pad each detected speech region by 400 ms so
                    # trailing consonants ('-ing', '-ed') and leading
                    # hard letters aren't clipped at the segment edge
                    # -- those clips are what makes whisper drop the
                    # tail of a word or hear 'thin' instead of 'thing'.
                    "speech_pad_ms": 400,
                    "threshold": 0.5,
                },
                without_timestamps=True,
                no_speech_threshold=0.6,
                compression_ratio_threshold=2.4,
                log_prob_threshold=-1.0,
                hotwords=self._hotwords,
                repetition_penalty=1.1,
            )
            tokens: List[str] = []
            for seg in segments:
                text = (seg.text or "").strip()
                if not text:
                    continue
                tokens.extend(text.split())
            return tokens

    def stream(
        self,
        *,
        stop_event,
        event_callback: Callable[[DictationEvent], None],
    ) -> bool:
        if not self._available:
            event_callback(DictationEvent(event="error", text=self._message))
            return False

        try:
            self._ensure_model()
        except Exception as exc:
            msg = f"model load failed: {exc}"
            self._message = msg
            print(f"[whisper-stream] {msg}")
            event_callback(DictationEvent(event="error", text=msg))
            return False

        audio_q: "queue.Queue[np.ndarray]" = queue.Queue()

        def _audio_cb(indata, frames, time_info, status):
            if status:
                print(f"[whisper-stream] capture status: {status}")
            audio_q.put(indata.reshape(-1).astype(np.float32, copy=True))

        stream_kwargs = dict(
            samplerate=_SAMPLE_RATE,
            channels=1,
            dtype="float32",
            blocksize=_BLOCK_SAMPLES,
            callback=_audio_cb,
        )
        if self._mic_index is not None:
            stream_kwargs["device"] = self._mic_index

        try:
            input_stream = sd.InputStream(**stream_kwargs)
            input_stream.start()
        except Exception as exc:
            msg = f"mic open failed: {exc}"
            print(f"[whisper-stream] {msg}")
            event_callback(DictationEvent(event="error", text=msg))
            return False

        event_callback(DictationEvent(event="ready"))
        mic_label = f"idx {self._mic_index}" if self._mic_index is not None else "default"
        print(f"[whisper-stream] listening (mic={mic_label}, backend={self._backend})")

        utterance = np.zeros(0, dtype=np.float32)
        silence_run_ms = 0.0
        speech_ms = 0.0

        try:
            while not stop_event.is_set():
                try:
                    first_block = audio_q.get(timeout=0.1)
                except queue.Empty:
                    continue
                # drain any backlog so we never fall behind
                blocks = [first_block]
                while True:
                    try:
                        blocks.append(audio_q.get_nowait())
                    except queue.Empty:
                        break
                block = np.concatenate(blocks) if len(blocks) > 1 else blocks[0]

                rms = float(np.sqrt(np.mean(block * block))) if block.size else 0.0
                is_silent_block = rms < _RMS_SILENCE_THRESHOLD
                block_ms = (block.size * 1000.0) / _SAMPLE_RATE

                # skip leading silence (don't start an utterance on room tone)
                if utterance.size == 0 and is_silent_block:
                    continue

                if utterance.size == 0:
                    silence_run_ms = 0.0
                    speech_ms = 0.0

                utterance = np.concatenate([utterance, block])
                if is_silent_block:
                    silence_run_ms += block_ms
                else:
                    silence_run_ms = 0.0
                    speech_ms += block_ms

                utt_ms = (utterance.size * 1000.0) / _SAMPLE_RATE

                should_commit = silence_run_ms >= _SILENCE_COMMIT_MS or utt_ms >= _MAX_UTTERANCE_MS
                if not should_commit:
                    continue

                # Skip decode if the utterance is mostly silence. Without this,
                # whisper hallucinates on low-signal audio — with hotwords
                # enabled it can spiral into a 200+ token repetition loop.
                if speech_ms < _MIN_SPEECH_MS:
                    print(f"[whisper-stream] skip decode (speech={speech_ms:.0f}ms < {_MIN_SPEECH_MS}ms, audio={utt_ms:.0f}ms)")
                    utterance = np.zeros(0, dtype=np.float32)
                    silence_run_ms = 0.0
                    speech_ms = 0.0
                    continue

                t0 = time.monotonic()
                try:
                    tokens = self._transcribe(utterance)
                except Exception as exc:
                    print(f"[whisper-stream] transcribe error: {exc}")
                    tokens = []
                decode_ms = (time.monotonic() - t0) * 1000.0
                print(f"[whisper-stream] decode audio={utt_ms:.0f}ms speech={speech_ms:.0f}ms took={decode_ms:.0f}ms tokens={len(tokens)}")

                final_text = " ".join(tokens).strip()
                if final_text:
                    event_callback(DictationEvent(event="final", text=final_text, confidence=1.0))
                    print(f"[whisper-stream] final (decode={decode_ms:.0f}ms, audio={utt_ms:.0f}ms): {final_text!r}")
                utterance = np.zeros(0, dtype=np.float32)
                silence_run_ms = 0.0
                speech_ms = 0.0
        except Exception as exc:
            msg = f"stream loop error: {exc}"
            print(f"[whisper-stream] {msg}")
            event_callback(DictationEvent(event="error", text=msg))
            return False
        finally:
            try:
                input_stream.stop()
                input_stream.close()
            except Exception:
                pass
            event_callback(DictationEvent(event="stopped"))
            self._message = "faster-whisper stopped"
            print(f"[whisper-stream] stopped")

        return True


# ---------------------------------------------------------------------------
# Legacy helpers retained for whisper.cpp batch paths (whisper_refiner,
# whisper-cli subprocess users). The streaming dictation path above no longer
# uses any of these, but WhisperRefiner still shells out to whisper-cli.exe
# against the ggml builds under whisper_bundle/.
# ---------------------------------------------------------------------------

_ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")

_META_LINE_PATTERNS = (
    re.compile(r"^whisper_", re.IGNORECASE),
    re.compile(r"^main:", re.IGNORECASE),
    re.compile(r"^init:", re.IGNORECASE),
    re.compile(r"^system_info", re.IGNORECASE),
    re.compile(r"^processing", re.IGNORECASE),
    re.compile(r"^\[start\]", re.IGNORECASE),
    re.compile(r"^\[end\]", re.IGNORECASE),
    re.compile(r"^\[blank_audio\]\s*$", re.IGNORECASE),
    re.compile(r"^### Transcription", re.IGNORECASE),
    re.compile(r"^---"),
    re.compile(r"^ggml_", re.IGNORECASE),
    re.compile(r"^build:", re.IGNORECASE),
    re.compile(r"^log\s*_?", re.IGNORECASE),
    re.compile(r"^SDL_", re.IGNORECASE),
)


def _is_meta_line(line: str) -> bool:
    if not line:
        return True
    for pattern in _META_LINE_PATTERNS:
        if pattern.search(line):
            return True
    return False


def _candidate_whisper_roots() -> list[Path]:
    roots: list[Path] = []
    base = app_base_path()
    roots.append(base)
    here = Path(__file__).resolve()
    for parent in here.parents:
        if parent not in roots:
            roots.append(parent)
    env = os.getenv("HGR_WHISPER_CPP_ROOT", "").strip()
    if env:
        roots.insert(0, Path(env))
    home_candidate = Path.home() / "Documents" / "whisper.cpp"
    if home_candidate not in roots:
        roots.append(home_candidate)
    return roots


def _candidate_model_roots() -> list[Path]:
    roots: list[Path] = []
    env = os.getenv("HGR_WHISPER_MODEL_DIR", "").strip()
    if env:
        roots.append(Path(env))
    roots.append(Path.home() / "Documents" / "TouchlessVoiceModels")
    roots.append(Path.home() / "Documents" / "HGRVoiceModels")
    for root in _candidate_whisper_roots():
        roots.append(root / "models")
        roots.append(root / "whisper.cpp" / "models")
    return roots


def _first_existing_model(names: Iterable[str]) -> Optional[Path]:
    names_list = list(names)
    for root in _candidate_model_roots():
        if not root.exists():
            continue
        for name in names_list:
            path = root / name
            if path.exists():
                return path
    for root in _candidate_model_roots():
        if not root.exists():
            continue
        extras = sorted(p for p in root.glob("ggml-*.bin"))
        if extras:
            return extras[0]
    return None


def _detect_nvidia_gpu() -> bool:
    exe = shutil.which("nvidia-smi")
    if not exe:
        return False
    try:
        proc = subprocess.run(
            [exe, "-L"],
            capture_output=True,
            text=True,
            timeout=4.0,
            encoding="utf-8",
            errors="replace",
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
        )
    except (subprocess.SubprocessError, OSError):
        return False
    return proc.returncode == 0 and "GPU" in (proc.stdout or "")


def _detect_vulkan() -> bool:
    exe = shutil.which("vulkaninfo")
    if not exe:
        return False
    try:
        proc = subprocess.run(
            [exe, "--summary"],
            capture_output=True,
            text=True,
            timeout=4.0,
            encoding="utf-8",
            errors="replace",
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
        )
    except (subprocess.SubprocessError, OSError):
        return False
    if proc.returncode != 0:
        return False
    return "deviceName" in (proc.stdout or "") or "GPU" in (proc.stdout or "")


def _resolve_backend_executable(
    exe_name: str = "whisper-stream.exe",
) -> Optional[tuple[str, Path]]:
    override = os.getenv("HGR_WHISPER_BACKEND", "").strip().lower()
    backend_order: list[str]
    if sys.platform == "darwin":
        # macOS: the whisper.cpp Metal build (Apple GPU) is an extensionless
        # Mach-O in build_metal/. No CUDA/Vulkan on mac; CPU is the fallback.
        exe_name = exe_name[:-4] if exe_name.endswith(".exe") else exe_name
        backend_order = [override] if override in {"metal", "cpu"} else ["metal", "cpu"]
    elif override in {"cuda", "vulkan", "cpu"}:
        backend_order = [override]
    else:
        backend_order = []
        if _detect_nvidia_gpu():
            backend_order.append("cuda")
        if _detect_vulkan():
            backend_order.append("vulkan")
        backend_order.append("cpu")

    build_dirs = {
        "cuda": ("build_cuda",),
        "vulkan": ("build_vulkan",),
        "cpu": ("build_stream", "build_cpu"),
        "metal": ("build_metal",),
    }

    for backend in backend_order:
        for build in build_dirs[backend]:
            for root in _candidate_whisper_roots():
                for bundle in ("whisper_bundle", "whisper.cpp"):
                    for sub in ("bin/Release", "bin"):
                        candidate = root / bundle / build / sub / exe_name
                        if candidate.exists():
                            return backend, candidate
    return None

# Author: Konstantin Markov
