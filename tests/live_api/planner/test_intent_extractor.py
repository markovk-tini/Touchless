"""Tests for the Tier-1.5 local-LLM intent extractor.

Covers the design's fallback matrix:
    - Ollama-off short-circuit (fast-fail via is_available)
    - Valid extraction round-trip for a few tool shapes
    - Negative signals: {"tool": null}, hallucinated tool name,
      missing required arg, wrong type, out-of-band values
    - Cache: identical utterances hit cache on the second call;
      negative cache also holds
    - Markdown-fence stripping tolerates ```json ...``` wrappers
    - Substring-sanity: outlook_compose with an @-address not present
      in the raw utterance is rejected

Author: Konstantin Markov
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from hgr.live_api.planner.intent_extractor import IntentExtractor  # noqa: E402


# ---------- fake registry ---------------------------------------------

class _FakeRegistry:
    """Minimal ToolRegistry stand-in.

    - `is_available(name)` returns whatever `avail_reason` was set to
      (None = available, str = short reason string).
    - `handles_connector(name)` returns True for every tool in
      `handles`.
    - `call("ollama_generate", ...)` returns the pre-scripted response.
    """

    def __init__(self,
                 responses: Optional[List[Dict[str, Any]]] = None,
                 avail_reason: Optional[str] = None,
                 handles: Optional[set] = None) -> None:
        self._responses: List[Dict[str, Any]] = list(responses or [])
        self._avail_reason = avail_reason
        self._handles = handles if handles is not None else {
            "ollama_generate", "sheets_create", "volume_set",
            "discord_mute", "outlook_compose", "gdocs_create",
            "tasks_add", "drive_upload",
        }
        self.calls: List[tuple] = []

    def is_available(self, name: str) -> Optional[str]:
        if name == "ollama_generate":
            return self._avail_reason
        return None

    def handles_connector(self, name: str) -> bool:
        return name in self._handles

    def call(self, tool: str,
             args: Dict[str, Any]) -> Dict[str, Any]:
        self.calls.append((tool, dict(args or {})))
        if tool == "ollama_generate":
            if not self._responses:
                return {"status": "error", "error": "no scripted response"}
            return self._responses.pop(0)
        return {"status": "ok"}


def _resp(payload: Any) -> Dict[str, Any]:
    """Wrap a JSON-serializable payload as an ollama_generate 'ok'
    response with .text == json-encoded payload."""
    return {"status": "ok", "text": json.dumps(payload)}


def _raw(text: str) -> Dict[str, Any]:
    """Return an ollama_generate 'ok' response with `text` verbatim."""
    return {"status": "ok", "text": text}


# ---------- 1. Ollama-off short-circuit -------------------------------

def test_ollama_off_short_circuits_without_calling_model():
    reg = _FakeRegistry(responses=[], avail_reason="ollama not connected")
    ex = IntentExtractor(reg)
    assert ex.try_extract("set volume to 50") is None
    # No ollama_generate call was ever issued.
    assert not any(t == "ollama_generate" for t, _ in reg.calls)


def test_handles_connector_off_short_circuits():
    reg = _FakeRegistry(responses=[], avail_reason=None,
                         handles=set())  # ollama not registered
    ex = IntentExtractor(reg)
    assert ex.try_extract("set volume to 50") is None
    assert not any(t == "ollama_generate" for t, _ in reg.calls)


# ---------- 2. Valid extraction round-trips ---------------------------

def test_sheets_create_valid_round_trip():
    payload = {
        "tool": "sheets_create",
        "args": {"title": "Q4 Budget",
                 "rows": [["name", "amount", "date"]]},
    }
    reg = _FakeRegistry(responses=[_resp(payload)])
    ex = IntentExtractor(reg)
    step = ex.try_extract(
        "spin up a google sheet called Q4 Budget with a header row "
        "for name amount and date")
    assert step is not None
    assert step.tool == "sheets_create"
    assert step.args["title"] == "Q4 Budget"
    assert step.args["rows"] == [["name", "amount", "date"]]
    assert step.layer == "connector"


def test_volume_set_valid_round_trip():
    payload = {"tool": "volume_set", "args": {"percent": 40}}
    reg = _FakeRegistry(responses=[_resp(payload)])
    ex = IntentExtractor(reg)
    step = ex.try_extract("crank the volume up to 40")
    assert step is not None
    assert step.tool == "volume_set"
    assert step.args["percent"] == 40


def test_discord_mute_valid_round_trip():
    payload = {"tool": "discord_mute", "args": {"muted": True}}
    reg = _FakeRegistry(responses=[_resp(payload)])
    ex = IntentExtractor(reg)
    step = ex.try_extract("silence me on discord")
    assert step is not None
    assert step.tool == "discord_mute"
    assert step.args["muted"] is True


# ---------- 3. Negative signals ---------------------------------------

def test_tool_null_returns_none():
    reg = _FakeRegistry(responses=[_resp({"tool": None})])
    ex = IntentExtractor(reg)
    assert ex.try_extract("what's the weather in Boston tomorrow") is None


def test_hallucinated_tool_name_rejected():
    reg = _FakeRegistry(responses=[
        _resp({"tool": "sheets_new", "args": {"title": "X"}}),
    ])
    ex = IntentExtractor(reg)
    assert ex.try_extract("make a sheet called X") is None


def test_missing_required_arg_rejected():
    reg = _FakeRegistry(responses=[
        _resp({"tool": "sheets_create", "args": {}}),
    ])
    ex = IntentExtractor(reg)
    assert ex.try_extract("make a sheet") is None


def test_wrong_type_rejected():
    reg = _FakeRegistry(responses=[
        _resp({"tool": "volume_set", "args": {"percent": "loud"}}),
    ])
    ex = IntentExtractor(reg)
    assert ex.try_extract("crank it up") is None


def test_out_of_band_percent_rejected():
    """percent=250 is out of the 0-100 bounds; the strict-safety
    policy from the design says miss, not clamp."""
    reg = _FakeRegistry(responses=[
        _resp({"tool": "volume_set", "args": {"percent": 250}}),
    ])
    ex = IntentExtractor(reg)
    assert ex.try_extract("crank volume way up") is None


def test_json_garbage_rejected():
    reg = _FakeRegistry(responses=[_raw("not json at all")])
    ex = IntentExtractor(reg)
    assert ex.try_extract("do a thing") is None


def test_tool_registered_but_connector_unregistered_rejected():
    """Model produces a corpus-valid tool the registry doesn't know
    about right now (e.g. Google Drive isn't authed)."""
    reg = _FakeRegistry(
        responses=[_resp({"tool": "sheets_create",
                          "args": {"title": "X"}})],
        handles={"ollama_generate"},  # sheets_create NOT registered
    )
    ex = IntentExtractor(reg)
    assert ex.try_extract("make a sheet titled X") is None


# ---------- 4. Cache: positive + negative -----------------------------

def test_positive_cache_hit_skips_second_ollama_call():
    payload = {"tool": "volume_set", "args": {"percent": 40}}
    reg = _FakeRegistry(responses=[_resp(payload)])
    ex = IntentExtractor(reg)
    utt = "crank the volume up to 40"
    step1 = ex.try_extract(utt)
    step2 = ex.try_extract(utt)
    assert step1 is not None
    assert step2 is not None
    # Only ONE ollama_generate invocation across both calls.
    ollama_calls = [t for t, _ in reg.calls if t == "ollama_generate"]
    assert len(ollama_calls) == 1


def test_negative_cache_hit_skips_second_ollama_call():
    reg = _FakeRegistry(responses=[_resp({"tool": None})])
    ex = IntentExtractor(reg)
    utt = "what's the weather"
    assert ex.try_extract(utt) is None
    assert ex.try_extract(utt) is None
    ollama_calls = [t for t, _ in reg.calls if t == "ollama_generate"]
    assert len(ollama_calls) == 1


# ---------- 5. Markdown fence stripping -------------------------------

def test_markdown_fenced_json_parses():
    fenced = ("```json\n"
              "{\"tool\": \"volume_set\", \"args\": {\"percent\": 25}}"
              "\n```")
    reg = _FakeRegistry(responses=[_raw(fenced)])
    ex = IntentExtractor(reg)
    step = ex.try_extract("volume 25")
    assert step is not None
    assert step.tool == "volume_set"
    assert step.args["percent"] == 25


def test_bare_fence_no_language_tag_parses():
    fenced = "```\n{\"tool\":\"discord_mute\",\"args\":{\"muted\":true}}\n```"
    reg = _FakeRegistry(responses=[_raw(fenced)])
    ex = IntentExtractor(reg)
    step = ex.try_extract("mute me")
    assert step is not None
    assert step.tool == "discord_mute"


# ---------- 6. Substring sanity ---------------------------------------

def test_outlook_compose_hallucinated_recipient_rejected():
    """Model emits a valid-shaped call but the recipient string
    doesn't appear in the raw utterance — should be rejected as
    hallucination."""
    payload = {
        "tool": "outlook_compose",
        "args": {"recipient": "fake@nowhere.com",
                 "body": "hi there"},
    }
    reg = _FakeRegistry(responses=[_resp(payload)])
    ex = IntentExtractor(reg)
    step = ex.try_extract("draft a quick email")
    assert step is None


def test_outlook_compose_real_recipient_accepted():
    payload = {
        "tool": "outlook_compose",
        "args": {"recipient": "dani@mangollc.org",
                 "body": "the Q3 report is done"},
    }
    reg = _FakeRegistry(responses=[_resp(payload)])
    ex = IntentExtractor(reg)
    step = ex.try_extract(
        "draft an email to dani@mangollc.org saying the "
        "Q3 report is done")
    assert step is not None
    assert step.tool == "outlook_compose"
    assert step.args["recipient"] == "dani@mangollc.org"
