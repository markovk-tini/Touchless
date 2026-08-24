"""Runtime configuration for the experimental Live API subsystem.

All values are read from environment variables when possible so the
prototype can be tweaked without code changes. The defaults are
intentionally conservative — see the README/feature spec for the full
list. Nothing here reaches into `AppConfig` or settings.json; the Live
API mode is a self-contained prototype.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional


# Model name is exposed as a constant so prototypes can swap it without
# editing dataclass defaults. "gpt-realtime" is the GA Realtime model.
# Overridable at runtime via OPENAI_REALTIME_MODEL.
DEFAULT_REALTIME_MODEL = "gpt-realtime"
DEFAULT_REALTIME_URL = "wss://api.openai.com/v1/realtime"

# Audio capture defaults — pcm16 mono is what the Realtime API expects.
DEFAULT_AUDIO_SAMPLE_RATE = 24000
DEFAULT_AUDIO_CHUNK_MS = 40  # ~960 frames at 24 kHz

# Screen-capture throttling.
SCREEN_CAPTURE_INTERVAL_SEC = 3.0
SCREEN_CAPTURE_MAX_WIDTH = 1280
SCREEN_CAPTURE_JPEG_QUALITY = 70


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on", "y"}


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


@dataclass
class LiveApiConfig:
    """User-tunable configuration for the Live API session."""

    # Connection
    api_key: Optional[str] = None
    model: str = DEFAULT_REALTIME_MODEL
    realtime_url: str = DEFAULT_REALTIME_URL

    # Voice used when Iris speaks (OpenAI Realtime voice id). Override
    # via LIVE_API_VOICE env var or the in-app Voice picker (which
    # mutates this field at runtime + persists to QSettings).
    # Default: 'marin' — released 2025 as a more consistent replacement
    # for sage/coral/verse, which drift in tone/gender mid-session.
    voice: str = "marin"

    # Feature gate — must be True for the manager to even attempt to start.
    enabled: bool = True

    # Audio
    audio_sample_rate: int = DEFAULT_AUDIO_SAMPLE_RATE
    audio_chunk_ms: int = DEFAULT_AUDIO_CHUNK_MS

    # Screen. On-demand by default: the model only receives a screenshot
    # at session start, after a screen-changing tool, or when it calls
    # get_screen_context — instead of streaming one every few seconds.
    # Images are the dominant token cost, so streaming was very wasteful.
    # Set LIVE_API_SEND_SCREEN_ALWAYS=true to restore continuous capture.
    send_screen_always: bool = False
    send_screen_interval_sec: float = SCREEN_CAPTURE_INTERVAL_SEC
    screen_max_width: int = SCREEN_CAPTURE_MAX_WIDTH
    screen_jpeg_quality: int = SCREEN_CAPTURE_JPEG_QUALITY

    # Logging / privacy
    # Default lives under ~/Documents/Touchless/ so:
    #   * the installed app can write here without admin (per-user install
    #     puts the .exe under %LOCALAPPDATA%\Programs\Touchless which IS
    #     writable, but logs there get blown away on every auto-update)
    #   * the path is the same in source-mode (`python run_app.py`) and
    #     installed-mode, so the user always knows where to look
    #   * survives auto-updates and re-installs
    log_dir: Path = field(
        default_factory=lambda: Path.home() / "Documents" / "Touchless" / "logs" / "live_api"
    )
    debug_text_logging: bool = False
    debug_save_screenshots: bool = False

    # Safety
    safe_workspace_dir: Path = field(
        default_factory=lambda: Path.home() / "Documents" / "Touchless" / "live_api_workspace"
    )

    # Misc
    reconnect_max_attempts: int = 3
    reconnect_backoff_sec: float = 2.0

    # ---- backend selection ----
    # Which backend powers the assistant:
    #   "cloud"        — OpenAI Realtime API; needs OPENAI_API_KEY.
    #                     Best quality / latency. Costs per request.
    #   "local"        — whisper.cpp + llama-server bundled with the
    #                     installer. 100% offline. No key. Lower
    #                     quality than gpt-realtime but zero marginal
    #                     cost; what a free-tier user gets.
    #   "subscription" — hosted Touchless proxy (planned). User auths
    #                     with their Touchless account; the proxy pays
    #                     for and rate-limits the upstream provider.
    #                     Today this is a stub that returns a clear
    #                     "subscription not yet available" error so the
    #                     UI can be built around it.
    #   "auto"         — pick the best available without user input.
    #                     OPENAI_API_KEY present → cloud.
    #                     Otherwise, local backend ready → local.
    #                     Otherwise → cloud (and surface a clear error
    #                     about the missing key, instead of silently
    #                     failing to start).
    # Default is "auto" because the shipped app should "just work" for
    # a user who installed it without any setup.
    backend: str = "auto"
    # GGUF model the local backend looks for. Default = Qwen 2.5 3B
    # Instruct (~2 GB) because that's also what the dictation grammar
    # corrector uses, so one downloaded file serves both features —
    # zero extra disk for users who already use dictation. The local
    # backend's PREFERRED_MODEL_FILES cascades to the 7B variant if
    # the user has placed that one instead.
    local_llm_model_filename: str = "Qwen2.5-3B-Instruct-Q4_K_M.gguf"
    # llama-server context window. Increased over the grammar-corrector's
    # 4096 because tool schemas + screen captions eat a lot of tokens.
    local_llm_context_size: int = 8192
    # Energy-based VAD threshold (RMS, 0..1). Above this counts as speech.
    local_vad_rms_threshold: float = 0.012
    # Silence required to end a turn (seconds). Mirrors the cloud
    # turn_detection.silence_duration_ms but for local audio.
    local_vad_silence_seconds: float = 1.5
    # Drop turns shorter than this (filters coughs/clicks).
    local_vad_min_speech_seconds: float = 0.4
    # Hard cap to flush a turn even if VAD never sees silence.
    local_vad_max_turn_seconds: float = 30.0
    # Max tool-call hops per user turn. Stops runaway loops where the
    # model keeps calling tools without producing a final message.
    local_max_tool_hops: int = 8


def load_config() -> LiveApiConfig:
    """Build a LiveApiConfig from environment + defaults.

    Called from LiveApiManager at startup. We do *not* cache — the user
    can change env vars and click Test Live API again to pick them up.
    """
    cfg = LiveApiConfig()
    cfg.api_key = os.environ.get("OPENAI_API_KEY") or None
    cfg.model = os.environ.get("OPENAI_REALTIME_MODEL", cfg.model).strip() or cfg.model
    cfg.realtime_url = os.environ.get("OPENAI_REALTIME_URL", cfg.realtime_url).strip() or cfg.realtime_url
    cfg.voice = (os.environ.get("LIVE_API_VOICE", cfg.voice).strip() or cfg.voice)

    cfg.enabled = _env_bool("TOUCHLESS_LIVE_API_ENABLED", cfg.enabled)
    cfg.audio_sample_rate = _env_int("LIVE_API_AUDIO_SAMPLE_RATE", cfg.audio_sample_rate)
    cfg.audio_chunk_ms = _env_int("LIVE_API_AUDIO_CHUNK_MS", cfg.audio_chunk_ms)

    cfg.send_screen_always = _env_bool("LIVE_API_SEND_SCREEN_ALWAYS", cfg.send_screen_always)
    cfg.send_screen_interval_sec = _env_float(
        "LIVE_API_SEND_SCREEN_INTERVAL_SEC", cfg.send_screen_interval_sec
    )
    cfg.screen_max_width = _env_int("LIVE_API_SCREEN_MAX_WIDTH", cfg.screen_max_width)
    cfg.screen_jpeg_quality = _env_int("LIVE_API_SCREEN_JPEG_QUALITY", cfg.screen_jpeg_quality)

    cfg.debug_text_logging = _env_bool("LIVE_API_DEBUG_TEXT_LOGGING", cfg.debug_text_logging)
    cfg.debug_save_screenshots = _env_bool("LIVE_API_DEBUG_SAVE_SCREENSHOTS", cfg.debug_save_screenshots)

    safe_ws = os.environ.get("LIVE_API_SAFE_WORKSPACE")
    if safe_ws:
        cfg.safe_workspace_dir = Path(safe_ws).expanduser()

    log_dir = os.environ.get("LIVE_API_LOG_DIR")
    if log_dir:
        cfg.log_dir = Path(log_dir).expanduser()

    backend_env = os.environ.get("TOUCHLESS_LIVE_API_BACKEND", "").strip().lower()
    if backend_env in {"cloud", "local", "subscription", "auto"}:
        cfg.backend = backend_env
    cfg.local_llm_model_filename = os.environ.get(
        "LIVE_API_LOCAL_MODEL", cfg.local_llm_model_filename
    )
    cfg.local_llm_context_size = _env_int(
        "LIVE_API_LOCAL_CTX", cfg.local_llm_context_size
    )
    cfg.local_vad_silence_seconds = _env_float(
        "LIVE_API_LOCAL_VAD_SILENCE_SEC", cfg.local_vad_silence_seconds
    )
    cfg.local_vad_rms_threshold = _env_float(
        "LIVE_API_LOCAL_VAD_RMS", cfg.local_vad_rms_threshold
    )

    # Resolve "auto" into a concrete backend so downstream code never
    # has to know about it. Done AFTER all env overrides apply so a
    # user-set TOUCHLESS_LIVE_API_BACKEND="auto" still goes through
    # the same selection logic.
    if (cfg.backend or "").strip().lower() == "auto":
        status = resolve_backend(cfg)
        cfg.backend = status["chosen"]
        # Stash the selection reason so the UI can surface it (e.g.
        # "auto-selected local because no API key was set"). Not part
        # of the dataclass schema; consumers tolerate setattr.
        try:
            cfg.backend_auto_reason = status["reason"]  # type: ignore[attr-defined]
        except Exception:
            pass

    return cfg


# ---- backend selection -----------------------------------------------------


def resolve_backend(cfg: "LiveApiConfig") -> Dict[str, Any]:
    """Decide which backend `cfg` should actually use right now.

    Priority order (matches the docstring on `backend = "auto"`):
      1. Cloud, when OPENAI_API_KEY is set — best quality.
      2. Subscription, when a Touchless subscription token is present
         (TOUCHLESS_SUBSCRIPTION_TOKEN env var). Stubbed today; the
         backend itself returns a clear error until the proxy ships.
      3. Local, when the bundled llama-server + GGUF model both resolve.
         The 100%-offline fallback every shipped user gets for free.
      4. Cloud (fallback), so the caller still surfaces the missing-key
         error to the UI in a recognisable shape rather than mysterious
         silence.

    Returns {chosen: str, reason: str, local: {ready, reason, ...}}.
    Safe to call with any cfg state; never raises."""
    has_cloud_key = bool((cfg.api_key or "").strip())
    has_subscription = bool(
        (os.environ.get("TOUCHLESS_SUBSCRIPTION_TOKEN") or "").strip())
    local_info: Dict[str, Any] = {"ready": False, "reason": "not probed"}
    try:
        from .local_backend import probe_local_backend
        local_info = probe_local_backend(
            model_filename=cfg.local_llm_model_filename,
            require_audio=False,  # text chat works without whisper
        )
    except Exception as exc:
        local_info = {"ready": False,
                      "reason": f"probe failed: {type(exc).__name__}: {exc}"}

    if has_cloud_key:
        chosen, reason = "cloud", "OPENAI_API_KEY set — using cloud"
    elif has_subscription:
        chosen, reason = ("subscription",
                          "Touchless subscription token detected")
    elif local_info.get("ready"):
        chosen, reason = ("local",
                          "no API key set; bundled local model ready")
    else:
        chosen, reason = (
            "cloud",
            "no API key, no subscription, local model unavailable "
            f"({local_info.get('reason')}). Cloud will surface the "
            "missing-key error.")
    return {"chosen": chosen, "reason": reason, "local": local_info}


# Author: Konstantin Markov
