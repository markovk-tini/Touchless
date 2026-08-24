"""Optional Claude Vision dispatch for rich screen queries.

Phase-4. The cheap ScreenReader path (UIA + DOM + OCR fallback)
covers ~80% of "what's on my screen" questions deterministically.
The remaining 20% — image-heavy content, charts, diagrams, hand-
written notes, video thumbnails — need a real vision model.

This module wraps Claude Sonnet's vision capability as an opt-in,
cost-gated dispatcher:

  * Requires `TOUCHLESS_VISION=1` env (off by default — vision
    calls cost real money).
  * Requires `ANTHROPIC_API_KEY` (no fallback to OpenAI yet —
    Phase-4-late; the multi-provider router will route this).
  * Consults the CostMeter via ModelRouter; falls through to
    None when over cap.
  * Captures the screen via `screen_context.py`'s existing
    screenshot helper rather than rebuilding it.
  * Returns a plain text answer; caller decides what to do with it.

Intentionally separate from `screen_awareness.py` — that module
runs ambient on a tight token budget; this one runs ON DEMAND
when the user asks a question the cheap path can't answer.

Author: Konstantin Markov
"""
from __future__ import annotations

import base64
import json
import os
import urllib.error
import urllib.request
from typing import Any, Optional, Tuple


_API_URL = "https://api.anthropic.com/v1/messages"
DEFAULT_MODEL = "claude-sonnet-4-6"
DEFAULT_MAX_TOKENS = 800
DEFAULT_TIMEOUT_S = 15.0


def vision_available() -> bool:
    """True iff vision dispatch is configured AND cost-gated open."""
    if os.environ.get("TOUCHLESS_VISION", "0") != "1":
        return False
    if not (os.environ.get("ANTHROPIC_API_KEY") or "").strip():
        return False
    # Cost cap check.
    try:
        from .cost_meter import global_meter
        if global_meter().is_over_cap():
            return False
    except Exception:
        pass
    return True


def ask_about_screen(question: str, *,
                     model: Optional[str] = None,
                     png_bytes: Optional[bytes] = None,
                     timeout_s: float = DEFAULT_TIMEOUT_S
                     ) -> Optional[str]:
    """Send a single vision query: 'here is the user's screen, here
    is their question, answer briefly'. Returns None on any failure
    (config off, network, JSON parse) so the caller can fall back
    to the cheap path.

    `png_bytes` may be supplied for testing; in production the
    function calls the project's existing screenshot helper."""
    if not vision_available():
        return None
    if not question:
        return None
    if png_bytes is None:
        try:
            png_bytes = _capture_screen_png()
        except Exception:
            return None
        if not png_bytes:
            return None
    try:
        payload = _build_request(model or DEFAULT_MODEL,
                                 question, png_bytes)
        text = _post(payload, timeout_s=timeout_s)
    except Exception:
        return None
    if not text:
        return None
    # Cost meter charge.
    try:
        from .cost_meter import global_meter
        # Rough estimate: ~1500 input tokens (image + prompt),
        # output proportional to reply length. Anthropic's actual
        # usage block isn't worth parsing yet; this is order-of-
        # magnitude correct.
        global_meter().record(
            model or DEFAULT_MODEL,
            tokens_in=1500,
            tokens_out=max(50, len(text) // 4),
        )
    except Exception:
        pass
    return text


# ---- internals ---------------------------------------------------------


def _capture_screen_png() -> Optional[bytes]:
    """Best-effort PNG bytes of the active monitor. Uses whatever
    screenshot helper the existing project has; falls through to
    None on non-Windows / missing libs."""
    try:
        # mss is the most common Python screenshot lib + already in
        # Touchless's screen_context path. Optional dep — guard.
        import mss  # type: ignore
        import io
        with mss.mss() as sct:
            shot = sct.shot(mon=1, output="-")
            # Newer mss returns bytes when output="-"; older returns
            # a file path. Handle both.
            if isinstance(shot, (bytes, bytearray)):
                return bytes(shot)
            # Path → read the file.
            with open(shot, "rb") as fp:
                return fp.read()
    except Exception:
        pass
    return None


def _build_request(model: str, question: str,
                   png_bytes: bytes) -> dict:
    b64 = base64.standard_b64encode(png_bytes).decode("ascii")
    return {
        "model": model,
        "max_tokens": DEFAULT_MAX_TOKENS,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image",
                 "source": {"type": "base64",
                            "media_type": "image/png",
                            "data": b64}},
                {"type": "text",
                 "text": _wrap_question(question)},
            ],
        }],
    }


def _wrap_question(question: str) -> str:
    """Add a tight system-style instruction so the model gives
    a useful, conversational answer instead of describing the
    whole image."""
    return (
        "Look at this screenshot. Answer the user's question "
        "briefly and naturally — like a smart friend reading "
        "their screen aloud. Don't describe the whole image; "
        "just answer.\n\n"
        f"Question: {question.strip()}"
    )


def _post(payload: dict, *, timeout_s: float) -> Optional[str]:
    key = (os.environ.get("ANTHROPIC_API_KEY") or "").strip()
    if not key:
        return None
    req = urllib.request.Request(
        _API_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "x-api-key": key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        # 429 → push the rate-limit signal so dependent paths know.
        if exc.code == 429:
            try:
                from .planner.scheduler import scheduler
                scheduler().record_rate_limit("anthropic")
            except Exception:
                pass
        return None
    except Exception:
        return None
    # Extract the first text block from the response content.
    try:
        blocks = data.get("content") or []
        for b in blocks:
            if b.get("type") == "text":
                return (b.get("text") or "").strip()
    except Exception:
        pass
    return None
