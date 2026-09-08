"""Tests for anthropic_client (Phase 4 B3)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import hgr.live_api.anthropic_client as ac  # noqa: E402
from hgr.live_api.anthropic_client import (  # noqa: E402
    call_messages, configured, messages_with_system_cache,
    record_spend,
)


# ---- env gating --------------------------------------------------------

def test_configured_requires_api_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert configured() is False
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    assert configured() is True


def test_call_returns_none_when_unconfigured(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    text, usage = call_messages(
        model="claude-haiku-4-5-20251001",
        messages=[{"role": "user", "content": "hi"}])
    assert text is None
    assert usage is None


# ---- request shape via patched urlopen --------------------------------

class _FakeResp:
    def __init__(self, body_bytes):
        self._b = body_bytes

    def read(self):
        return self._b

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _install_fake_urlopen(monkeypatch, response_payload):
    captured = {}

    def fake_urlopen(req, timeout=None):
        captured["url"] = req.full_url
        captured["headers"] = {k.lower(): v
                                for k, v in req.headers.items()}
        captured["body"] = json.loads(req.data.decode("utf-8"))
        captured["timeout"] = timeout
        return _FakeResp(json.dumps(response_payload).encode("utf-8"))

    monkeypatch.setattr(ac.urllib.request, "urlopen", fake_urlopen)
    return captured


def test_call_sends_anthropic_headers(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    cap = _install_fake_urlopen(monkeypatch, {
        "content": [{"type": "text", "text": "hello back"}],
        "usage": {"input_tokens": 10, "output_tokens": 5},
    })
    text, usage = call_messages(
        model="claude-haiku-4-5-20251001",
        messages=[{"role": "user", "content": "hi"}])
    assert text == "hello back"
    assert usage == {"input_tokens": 10, "output_tokens": 5}
    # Sanity-check the request shape.
    assert "anthropic.com" in cap["url"]
    assert cap["headers"]["x-api-key"] == "sk-test"
    assert cap["headers"]["anthropic-version"] == "2023-06-01"
    assert cap["body"]["model"] == "claude-haiku-4-5-20251001"


def test_call_with_system_string_attaches_to_top_level(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    cap = _install_fake_urlopen(monkeypatch, {
        "content": [{"type": "text", "text": "ok"}],
    })
    call_messages(model="x", system="You are Iris.",
                  messages=[{"role": "user", "content": "hi"}])
    body = cap["body"]
    # Anthropic API: system is a top-level field, NOT a message.
    assert body.get("system") == "You are Iris."
    assert all(m["role"] != "system" for m in body["messages"])


def test_call_with_system_blocks_preserves_cache_control(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    cap = _install_fake_urlopen(monkeypatch, {
        "content": [{"type": "text", "text": "ok"}],
    })
    sys_blocks = [
        {"type": "text", "text": "You are Iris.",
         "cache_control": {"type": "ephemeral"}},
        {"type": "text", "text": "TOOLS: ..."},
    ]
    call_messages(model="x", system=sys_blocks,
                  messages=[{"role": "user", "content": "hi"}])
    assert cap["body"]["system"] == sys_blocks


def test_call_json_mode_appends_prefill(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    cap = _install_fake_urlopen(monkeypatch, {
        "content": [{"type": "text", "text": "\"ok\": true}"}],
    })
    text, _ = call_messages(model="x",
                            messages=[{"role": "user", "content": "hi"}],
                            json_mode=True)
    # The prefill assistant message ('{') should appear as the last
    # message in the request body.
    body_msgs = cap["body"]["messages"]
    assert body_msgs[-1]["role"] == "assistant"
    assert body_msgs[-1]["content"] == "{"
    # Reply should be prepended with '{' if missing.
    assert text.startswith("{")


def test_call_http_429_records_anthropic_rate_limit(monkeypatch):
    from hgr.live_api.planner.scheduler import scheduler
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")

    import urllib.error

    def raise_429(req, timeout=None):
        raise urllib.error.HTTPError(
            url="x", code=429, msg="rate", hdrs=None, fp=None)

    monkeypatch.setattr(ac.urllib.request, "urlopen", raise_429)
    scheduler()._events.clear()  # reset
    text, usage = call_messages(model="x",
                                 messages=[{"role": "user",
                                            "content": "hi"}])
    assert text is None
    # Scheduler should know about the rate limit.
    assert scheduler().is_throttled("anthropic", 60.0)
    scheduler()._events.clear()  # cleanup


def test_call_returns_none_on_network_error(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")

    def fail(req, timeout=None):
        raise OSError("network down")

    monkeypatch.setattr(ac.urllib.request, "urlopen", fail)
    text, _ = call_messages(model="x",
                             messages=[{"role": "user", "content": "hi"}])
    assert text is None


# ---- helpers -----------------------------------------------------------

def test_messages_with_system_cache_uses_prompt_cache_helper():
    kwargs = messages_with_system_cache(
        system_prompt="You are Iris.",
        tool_catalog_text="TOOLS: ...",
        user_text="hello",
    )
    assert "system" in kwargs
    assert "messages" in kwargs
    # System blocks should be marked ephemeral for prompt caching.
    sys_blocks = kwargs["system"]
    assert isinstance(sys_blocks, list)
    assert all(b.get("cache_control") == {"type": "ephemeral"}
               for b in sys_blocks)


def test_record_spend_uses_usage_when_present(monkeypatch):
    charged = []
    class _Meter:
        def record(self, model, *, tokens_in, tokens_out):
            charged.append((model, tokens_in, tokens_out))
    import hgr.live_api.cost_meter as cm
    monkeypatch.setattr(cm, "global_meter", lambda: _Meter())
    record_spend("claude-haiku",
                 {"input_tokens": 100, "output_tokens": 50})
    assert charged == [("claude-haiku", 100, 50)]


def test_record_spend_falls_back_to_char_estimate(monkeypatch):
    charged = []
    class _Meter:
        def record(self, model, *, tokens_in, tokens_out):
            charged.append((model, tokens_in, tokens_out))
    import hgr.live_api.cost_meter as cm
    monkeypatch.setattr(cm, "global_meter", lambda: _Meter())
    # No usage block — falls back to char/4 estimate.
    record_spend("claude-haiku", None,
                 fallback_in_chars=400, fallback_out_chars=200)
    assert len(charged) == 1
    _model, tin, tout = charged[0]
    assert tin == 100  # 400 / 4
    assert tout == 50   # 200 / 4
