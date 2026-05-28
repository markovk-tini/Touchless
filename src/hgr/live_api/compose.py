"""compose_text — mid-plan natural-language synthesis.

The {step:N.field} ref system can't iterate or summarize — it can only do
direct string substitution. For requests like 'write a debrief about my
emails into a doc', the planner needs to turn structured step outputs
(weather dict, list of emails) into prose BEFORE the next step (e.g.
gdocs_create) consumes it as a `text` arg.

compose_text takes a prompt + arbitrary input data (resolved from
{step:N.field} refs), calls a cheap-LLM, and returns {"text": "..."}
so the next step can use {step:N.text}.

Same backing infrastructure as the Synthesizer (one cheap-LLM call,
gpt-5-mini default, ~500 tokens, ~$0.0001 per call).

Author: Konstantin Markov
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any, Dict

DEFAULT_MODEL = "gpt-5-mini"
API_URL = "https://api.openai.com/v1/chat/completions"
_MAX_INPUT_CHARS = 6000   # cap to keep prompt small
_MAX_TOKENS = 500          # cap output length


def configured() -> bool:
    return bool((os.environ.get("OPENAI_API_KEY") or "").strip())


def compose_text(prompt: str, inputs: Any, max_tokens: int = _MAX_TOKENS,
                 model: str = "") -> Dict[str, Any]:
    """Synthesize text from a prompt + inputs.

    `inputs` can be a string (e.g. pre-formatted "weather: ..., emails: ...")
    or a dict (any JSON-serializable structure — typically the resolved
    output of prior steps). Returns {status, text} on success.
    """
    if not prompt:
        return {"status": "error", "error": "prompt is required",
                "code": "invalid_arguments"}
    if not configured():
        return {"status": "error", "error": "OPENAI_API_KEY not set",
                "code": "not_configured"}

    chosen_model = (model
                    or os.environ.get("TOUCHLESS_COMPOSE_MODEL")
                    or os.environ.get("TOUCHLESS_PLANNER_MODEL")
                    or DEFAULT_MODEL)

    if not isinstance(inputs, str):
        try:
            inputs_str = json.dumps(inputs, default=str)
        except Exception:
            inputs_str = str(inputs)
    else:
        inputs_str = inputs
    inputs_str = inputs_str[:_MAX_INPUT_CHARS]

    system = (
        "You are a text composer. Given the user's instruction and the "
        "input data, produce ONLY the final text. No preamble, no "
        "explanation, no JSON wrapping — just the text the user wants. "
        "Adapt length and structure to the instruction (a 'short summary' "
        "should be concise; a 'detailed brief' should expand). Enumerate "
        "every relevant item in lists when the user asked for a complete "
        "listing; don't truncate."
    )
    user = f"Instruction:\n{prompt}\n\nInput data:\n{inputs_str}"

    body = {
        "model": chosen_model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": 0.3,
        "max_tokens": int(max(64, min(int(max_tokens or _MAX_TOKENS), 2000))),
    }
    key = os.environ["OPENAI_API_KEY"]
    req = urllib.request.Request(
        API_URL, data=json.dumps(body).encode("utf-8"),
        headers={"Authorization": f"Bearer {key}",
                 "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return {"status": "error",
                "error": f"HTTP {exc.code}",
                "code": "http_error"}
    except Exception as exc:
        return {"status": "error",
                "error": f"{type(exc).__name__}: {exc}",
                "code": "compose_failed"}
    text = ((payload.get("choices") or [{}])[0]
            .get("message", {}).get("content") or "").strip()
    return {"status": "ok", "text": text, "model": chosen_model}
