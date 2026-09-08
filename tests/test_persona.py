"""Tests for persona anchor resolver (Phase 3)."""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import hgr.live_api.persona as persona_mod  # noqa: E402
from hgr.live_api.persona import (  # noqa: E402
    DEFAULT_PERSONA, MAX_PERSONA_CHARS, get_persona_block,
    reset_for_tests, set_persona_block,
)


def setup_function():
    reset_for_tests()
    # Phase-7: get_persona_block() now defers to persona_voice
    # presets when no explicit / env / file / memory override is
    # set. Reset persona_voice so the "default" preset is active
    # (and its style_block + few-shot examples are what we get).
    try:
        from hgr.live_api import persona_voice as _pv
        _pv.reset_for_tests()
    except Exception:
        pass


def _is_default_style(text: str) -> bool:
    """Phase-7: when nothing's set, get_persona_block() may return
    either the legacy DEFAULT_PERSONA string or the persona_voice
    'default' preset's style_block. Both encode the same vibe;
    accept either."""
    if text == DEFAULT_PERSONA:
        return True
    # The persona_voice default preset uses the same anchor words.
    needles = ("warmly", "dry wit", "Never read URLs")
    return all(n in text for n in needles)


# ---- default + explicit set --------------------------------------------

def test_default_used_when_nothing_set(monkeypatch):
    monkeypatch.delenv("TOUCHLESS_PERSONA", raising=False)
    monkeypatch.delenv("TOUCHLESS_PERSONA_FILE", raising=False)
    assert _is_default_style(get_persona_block())


def test_explicit_set_overrides_default(monkeypatch):
    monkeypatch.delenv("TOUCHLESS_PERSONA", raising=False)
    set_persona_block("Be terse. British English.")
    assert get_persona_block() == "Be terse. British English."


def test_explicit_set_to_none_falls_through_to_default(monkeypatch):
    monkeypatch.delenv("TOUCHLESS_PERSONA", raising=False)
    set_persona_block("anything")
    set_persona_block(None)
    assert _is_default_style(get_persona_block())


def test_explicit_set_clamped_to_max(monkeypatch):
    monkeypatch.delenv("TOUCHLESS_PERSONA", raising=False)
    huge = "x" * (MAX_PERSONA_CHARS + 100)
    set_persona_block(huge)
    assert len(get_persona_block()) == MAX_PERSONA_CHARS


# ---- env-var resolution -------------------------------------------------

def test_env_var_overrides_default(monkeypatch):
    monkeypatch.setenv("TOUCHLESS_PERSONA", "from env")
    assert get_persona_block() == "from env"


def test_explicit_beats_env(monkeypatch):
    monkeypatch.setenv("TOUCHLESS_PERSONA", "from env")
    set_persona_block("from set")
    assert get_persona_block() == "from set"


def test_env_file_overrides_default(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "persona.txt"
        p.write_text("from file", encoding="utf-8")
        monkeypatch.delenv("TOUCHLESS_PERSONA", raising=False)
        monkeypatch.setenv("TOUCHLESS_PERSONA_FILE", str(p))
        assert get_persona_block() == "from file"


def test_env_var_beats_file(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "persona.txt"
        p.write_text("from file", encoding="utf-8")
        monkeypatch.setenv("TOUCHLESS_PERSONA", "from env")
        monkeypatch.setenv("TOUCHLESS_PERSONA_FILE", str(p))
        assert get_persona_block() == "from env"


def test_env_file_missing_falls_through_to_default(monkeypatch):
    monkeypatch.delenv("TOUCHLESS_PERSONA", raising=False)
    monkeypatch.setenv("TOUCHLESS_PERSONA_FILE", "/does/not/exist")
    assert _is_default_style(get_persona_block())


# ---- memory-backed persona ----------------------------------------------

class _FakeFact:
    def __init__(self, value): self.value = value


class _FakeStore:
    def __init__(self, value): self._v = value

    def find_facts(self, *, kind, key):
        if kind == "preference" and key == "persona":
            return [_FakeFact(self._v)]
        return []


class _FakeMemory:
    def __init__(self, value): self._store = _FakeStore(value)


def test_memory_persona_used_when_no_explicit_no_env(monkeypatch):
    monkeypatch.delenv("TOUCHLESS_PERSONA", raising=False)
    monkeypatch.delenv("TOUCHLESS_PERSONA_FILE", raising=False)
    mem = _FakeMemory("be concise")
    assert get_persona_block(memory=mem) == "be concise"


def test_env_beats_memory(monkeypatch):
    monkeypatch.setenv("TOUCHLESS_PERSONA", "env override")
    mem = _FakeMemory("memory ver")
    assert get_persona_block(memory=mem) == "env override"


def test_memory_persona_clamped_to_max(monkeypatch):
    monkeypatch.delenv("TOUCHLESS_PERSONA", raising=False)
    monkeypatch.delenv("TOUCHLESS_PERSONA_FILE", raising=False)
    huge = "y" * (MAX_PERSONA_CHARS + 200)
    mem = _FakeMemory(huge)
    assert len(get_persona_block(memory=mem)) == MAX_PERSONA_CHARS
