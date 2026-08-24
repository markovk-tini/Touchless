"""Ollama connector — local LLM hosting via the Ollama HTTP API.

Lets Iris route small/private/bulk tasks to a model running on the user's
own machine — no API key, no rate limit, no data leaving the box — reserving
gpt-realtime for the multi-step reasoning that actually needs it. Massive
cost + latency + privacy win for summarization, classification, extraction,
translation, regex generation, simple Q&A.

`setup_self()` is the entry the iris_setup_tool calls for "set up ollama":
- already installed + running → reports the model list + picks a default
- installed but server not up → spawns `ollama serve` in the background
- not installed → returns the install URL and stops

No third-party deps — uses stdlib urllib + subprocess only. The Ollama
server uses `localhost` so there's no TLS / cert concern.

Author: Konstantin Markov
"""
from __future__ import annotations

import json
import shutil
import subprocess
import time
import urllib.error
import urllib.request
import webbrowser
from typing import Any, Dict, List, Optional

from .base import Connector, connector_result


# Default Ollama listen port. Override via the OLLAMA_HOST env var is honoured
# by the Ollama binary itself; we only ever talk to localhost.
OLLAMA_HOST = "http://localhost:11434"

# Cache the cheap "is the server up?" probe so the capability-search router
# doesn't pay an HTTP round-trip every time it walks the connector list.
_AVAIL_CACHE_TTL_SECONDS = 5.0

# Probe timeout: small enough that an absent server fails fast (the registry
# walks every connector's available() on each turn). Generate timeout: large
# enough for a small model's longer answers; user can extend by running a
# bigger task in chunks.
_HTTP_TIMEOUT_PROBE = 0.6
_HTTP_TIMEOUT_GENERATE = 120.0


def _ollama_get(path: str, timeout: float) -> Optional[Dict[str, Any]]:
    """GET a JSON endpoint from the local Ollama server. Returns None on any
    error (connection refused, timeout, non-200, bad JSON). The caller treats
    that as 'server unavailable' — never an exception."""
    try:
        req = urllib.request.Request(
            OLLAMA_HOST + path,
            headers={"Accept": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8", errors="replace"))
    except Exception:
        return None


def _ollama_post(path: str, body: Dict[str, Any],
                 timeout: float) -> Optional[Dict[str, Any]]:
    """POST JSON to the local Ollama server. Same None-on-error contract."""
    try:
        data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(
            OLLAMA_HOST + path,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8", errors="replace"))
    except Exception:
        return None


def _ollama_running() -> bool:
    """Cheap probe: does the local Ollama API respond? Used both for
    available() and to detect 'installed but not running' in setup_self."""
    return _ollama_get("/api/tags", _HTTP_TIMEOUT_PROBE) is not None


def _ollama_models() -> List[str]:
    """Names of locally-pulled models (in whatever order /api/tags returns —
    typically alphabetical). Empty list when the server isn't up."""
    data = _ollama_get("/api/tags", _HTTP_TIMEOUT_PROBE)
    if not data:
        return []
    raw = data.get("models") or []
    return [m.get("name", "") for m in raw if m.get("name")]


def _ollama_cli_path() -> Optional[str]:
    """Locate the `ollama` binary. Windows installers put it under
    %LOCALAPPDATA%\\Programs\\Ollama\\ollama.exe, which the installer adds to
    PATH — so shutil.which finds it. None means Ollama isn't installed."""
    return shutil.which("ollama")


# Models we prefer as a default when the user has multiple installed. Small,
# fast, generally-capable models float to the top; coding-specific or huge
# models stay at the bottom unless they're the only thing installed. The
# entries are SUBSTRINGS — 'llama3.2' matches 'llama3.2:latest', etc.
_DEFAULT_MODEL_PREFERENCE: tuple = (
    "qwen2.5:3b", "qwen2.5", "llama3.2:3b", "llama3.2",
    "phi3.5", "phi3", "mistral", "gemma2:2b", "gemma2",
)


def _pick_default_model(models: List[str]) -> Optional[str]:
    """Choose a sensible default from what's installed. Picks the smallest /
    most-general purpose model available so the first generation doesn't take
    forever loading a 70B into VRAM."""
    if not models:
        return None
    low = [(m, m.lower()) for m in models]
    for pref in _DEFAULT_MODEL_PREFERENCE:
        for original, l in low:
            if pref in l:
                return original
    # Nothing preferred — return whatever's first.
    return models[0]


class OllamaConnector(Connector):
    """Local LLM via Ollama. Free, fast, private, no rate limit.

    Tools exposed:
      ollama_generate(prompt, system?, model?) -> {text, model, ...}
      ollama_list_models() -> {models, default}

    Multi-turn chat is intentionally NOT exposed yet — gpt-realtime IS the
    chat. Ollama is for "process this once and give me a result," not for
    holding a parallel conversation. Add a chat tool later if a real
    use case turns up.
    """

    id = "ollama"
    description = (
        "Local LLM (Llama, Qwen, DeepSeek, Mistral, Phi, Gemma, etc.) via "
        "Ollama running on the user's machine. Text generation, "
        "summarization, classification, translation, extraction, regex, "
        "simple Q&A — free, fast, private, no rate limit. Prefer this over "
        "gpt-realtime for SMALL or BULK or PRIVATE tasks that don't need "
        "vision, multi-step reasoning, or recent-world knowledge."
    )

    def __init__(self) -> None:
        self._available_until: float = 0.0
        self._cached_available: bool = False
        self._default_model: Optional[str] = None

    # ---- setup --------------------------------------------------------------

    def setup_self(self) -> Dict[str, Any]:
        """Detect Ollama and report a user-friendly status. Three outcomes:
        (1) already running → ok + model list + default; (2) installed but
        not running → spawn `ollama serve` and re-probe; (3) not installed →
        return the install URL so the model can tell the user what to do."""
        # Case 1: already up.
        if _ollama_running():
            return self._ok_with_models()

        # Case 2: installed but server not up — spawn it.
        cli = _ollama_cli_path()
        if cli:
            try:
                # No console window, no inherited stdio — the serve process
                # runs detached in the background until the user reboots /
                # quits Ollama from the system tray.
                subprocess.Popen(
                    [cli, "serve"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    stdin=subprocess.DEVNULL,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                )
            except Exception as exc:
                return {
                    "ok": False,
                    "error": (f"Found Ollama at {cli} but couldn't start "
                              f"the server: {type(exc).__name__}: {exc}. "
                              "Try opening Ollama from the Start menu."),
                }
            # Give it a few seconds to come up.
            deadline = time.time() + 6.0
            while time.time() < deadline:
                if _ollama_running():
                    return self._ok_with_models()
                time.sleep(0.3)
            return {
                "ok": False,
                "error": ("Found Ollama but the local server didn't come up "
                          "within a few seconds. Try opening Ollama from "
                          "the Start menu, or run `ollama serve` in a "
                          "terminal."),
            }

        # Case 3: not installed. Open the download page in the user's browser
        # so they don't have to copy a URL — matches the Notion connector's UX.
        try:
            webbrowser.open("https://ollama.com/download")
        except Exception:
            pass
        return {
            "ok": False,
            "error": ("Ollama isn't installed. I've opened "
                      "https://ollama.com/download in your browser "
                      "(free, ~200 MB). After install, pull a small "
                      "model — `ollama pull qwen2.5:3b` (2 GB) is a good "
                      "first default — then ask me to set up Ollama again."),
        }

    def _ok_with_models(self) -> Dict[str, Any]:
        """Build the success payload for setup_self — also primes our model
        cache so the first ollama_generate doesn't pay a probe round-trip."""
        models = _ollama_models()
        if not models:
            # Server up, no models pulled.
            return {
                "ok": True,
                "running": True,
                "models": [],
                "message": ("Ollama is running but has no models. Pull one "
                            "with `ollama pull qwen2.5:3b` (2 GB, small + "
                            "fast) or `ollama pull llama3.2` (2 GB) for a "
                            "good default."),
            }
        default = _pick_default_model(models)
        self._default_model = default
        self._cached_available = True
        self._available_until = time.time() + _AVAIL_CACHE_TTL_SECONDS
        return {
            "ok": True,
            "running": True,
            "models": models,
            "default_model": default,
            "message": (f"Ollama is connected with {len(models)} model"
                        f"{'s' if len(models) != 1 else ''} installed. "
                        f"Default for quick tasks: {default}."),
        }

    # ---- registry hooks -----------------------------------------------------

    def available(self) -> bool:
        """True when the local Ollama API responds AND at least one model is
        pulled. Cached for ~5s so the capability-search router doesn't pay an
        HTTP probe on every turn."""
        now = time.time()
        if now < self._available_until:
            return self._cached_available
        models = _ollama_models()
        ok = bool(models)
        self._cached_available = ok
        self._available_until = now + _AVAIL_CACHE_TTL_SECONDS
        if ok and not self._default_model:
            self._default_model = _pick_default_model(models)
        return ok

    def tools(self) -> List[Dict[str, Any]]:
        return [
            {
                "type": "function",
                "name": "ollama_generate",
                "description": (
                    "Run a one-shot text generation on the user's LOCAL "
                    "Ollama model — free, fast, private, no rate limit. "
                    "Use for summarization, classification, extraction, "
                    "translation, regex generation, simple Q&A — anything "
                    "that doesn't need vision, multi-step reasoning, or "
                    "recent-world knowledge. Returns the generated text. "
                    "If the user asks to process many items, call this in "
                    "a loop rather than burning gpt-realtime tokens."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "prompt": {
                            "type": "string",
                            "description": "The instruction or input text.",
                        },
                        "system": {
                            "type": "string",
                            "description": ("Optional system message setting "
                                            "the role/style (e.g. 'You are "
                                            "a concise summarizer')."),
                        },
                        "model": {
                            "type": "string",
                            "description": ("Optional model name (e.g. "
                                            "'qwen2.5:3b', 'llama3.2'). Omit "
                                            "to use the connector's default."),
                        },
                    },
                    "required": ["prompt"],
                    "additionalProperties": False,
                },
            },
            {
                "type": "function",
                "name": "ollama_list_models",
                "description": (
                    "List the LLM models pulled locally on this machine. Use "
                    "to pick the right model for a task or to tell the user "
                    "what they have available."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                    "additionalProperties": False,
                },
            },
        ]

    # ---- execution ----------------------------------------------------------

    def execute(self, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        if name == "ollama_list_models":
            models = _ollama_models()
            if not self._default_model and models:
                self._default_model = _pick_default_model(models)
            return connector_result(
                "ok",
                models=models,
                default=self._default_model,
            )

        if name == "ollama_generate":
            prompt = str(args.get("prompt") or "").strip()
            if not prompt:
                return connector_result(
                    "error", error="prompt is required",
                    code="invalid_arguments",
                )
            # Model: explicit arg → cached default → fresh probe → error.
            model = str(args.get("model") or "").strip() or self._default_model
            if not model:
                models = _ollama_models()
                if not models:
                    return connector_result(
                        "error",
                        error=("No Ollama models installed. Pull one with "
                               "`ollama pull qwen2.5:3b`."),
                        code="no_model",
                    )
                model = _pick_default_model(models) or models[0]
                self._default_model = model
            body: Dict[str, Any] = {
                "model": model,
                "prompt": prompt,
                # We always wait for the full answer — the realtime model
                # consumes whole results, never streams. Streaming would add
                # complexity for no caller benefit.
                "stream": False,
            }
            system = str(args.get("system") or "").strip()
            if system:
                body["system"] = system
            data = _ollama_post(
                "/api/generate", body, _HTTP_TIMEOUT_GENERATE
            )
            if data is None:
                return connector_result(
                    "error",
                    error=("Couldn't reach the Ollama server. Is it running? "
                           "Try `ollama serve` in a terminal, or open the "
                           "Ollama app from the Start menu."),
                    code="ollama_unreachable",
                )
            text = (data.get("response") or "").strip()
            # eval_duration is in nanoseconds per the Ollama API contract.
            eval_dur_ns = data.get("eval_duration")
            eval_ms = (eval_dur_ns // 1_000_000) if eval_dur_ns else None
            return connector_result(
                "ok",
                model=model,
                text=text,
                eval_count=data.get("eval_count"),
                eval_ms=eval_ms,
            )

        return connector_result(
            "error",
            error=f"unknown ollama tool: {name}",
            code="no_handler",
        )
