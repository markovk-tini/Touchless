"""Tests for persona_voice presets (Phase 7 B1)."""
from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import hgr.live_api.persona_voice as pv  # noqa: E402
from hgr.live_api.persona_voice import (  # noqa: E402
    PersonaPreset, active_preset, all_presets, get_preset,
    record_outcome, reset_for_tests, set_active, style_block,
    temperature, usage_snapshot,
)


def setup_function():
    reset_for_tests()


# ---- preset catalogue -----------------------------------------------

def test_default_preset_present():
    p = get_preset("default")
    assert p is not None
    assert p.name == "default"


def test_jarvis_preset_present():
    p = get_preset("jarvis")
    assert p is not None
    assert "sir" in p.style_block.lower()


def test_concise_preset_low_temperature():
    p = get_preset("concise")
    assert p is not None
    assert p.temperature <= 0.4


def test_all_presets_returns_list():
    presets = all_presets()
    assert len(presets) >= 4
    names = {p.name for p in presets}
    assert {"default", "jarvis", "concise", "warm"}.issubset(names)


def test_get_unknown_preset_returns_none():
    assert get_preset("nonexistent") is None


def test_get_preset_handles_blank():
    assert get_preset("") is None
    assert get_preset("   ") is None


def test_get_preset_case_insensitive():
    assert get_preset("JARVIS") is not None
    assert get_preset("Jarvis") is not None


# ---- active resolution ----------------------------------------------

def test_active_defaults_to_default():
    assert active_preset().name == "default"


def test_set_active_sticks():
    assert set_active("jarvis") is True
    assert active_preset().name == "jarvis"


def test_set_active_clear():
    set_active("warm")
    set_active(None)
    assert active_preset().name == "default"


def test_set_active_rejects_unknown():
    assert set_active("does-not-exist") is False
    assert active_preset().name == "default"


def test_env_var_resolves_when_no_override():
    with patch.dict(os.environ,
                    {"TOUCHLESS_PERSONA_PRESET": "playful"}):
        assert active_preset().name == "playful"


def test_override_beats_env():
    with patch.dict(os.environ,
                    {"TOUCHLESS_PERSONA_PRESET": "playful"}):
        set_active("jarvis")
        assert active_preset().name == "jarvis"


# ---- style block + temperature ---------------------------------------

def test_style_block_non_empty():
    block = style_block()
    assert block
    assert len(block) > 50


def test_style_block_includes_examples_by_default():
    block = style_block()
    assert "Example exchanges" in block


def test_style_block_excludes_examples_when_flagged():
    block = style_block(with_examples=False)
    assert "Example exchanges" not in block


def test_temperature_reflects_active_preset():
    set_active("concise")
    assert temperature() <= 0.4
    set_active("playful")
    assert temperature() >= 0.6


# ---- few-shot ----------------------------------------------------------

def test_few_shot_block_capped_by_max():
    p = get_preset("default")
    block = p.few_shot_block(max_examples=2)
    # 2 example pairs, each formatted "User: ... \nIris: ..."
    assert block.count("User:") == 2


def test_few_shot_block_empty_when_zero():
    p = get_preset("default")
    assert p.few_shot_block(max_examples=0) == ""


# ---- bandit / outcome tracking -------------------------------------

def test_record_outcome_kept_increments():
    set_active("jarvis")
    record_outcome("jarvis", kept=True)
    snap = usage_snapshot()
    assert snap["jarvis"]["kept_replies"] == 1
    assert snap["jarvis"]["revised"] == 0


def test_record_outcome_revised_increments():
    record_outcome("jarvis", kept=False)
    snap = usage_snapshot()
    assert snap["jarvis"]["revised"] == 1


def test_record_outcome_unknown_preset_ignored():
    record_outcome("nope", kept=True)
    snap = usage_snapshot()
    assert "nope" not in snap or snap["nope"]["kept_replies"] == 0


def test_keep_rate_computed():
    record_outcome("warm", kept=True)
    record_outcome("warm", kept=True)
    record_outcome("warm", kept=False)
    snap = usage_snapshot()
    assert snap["warm"]["keep_rate"] is not None
    assert abs(snap["warm"]["keep_rate"] - (2/3)) < 1e-6


def test_keep_rate_none_when_no_replies():
    snap = usage_snapshot()
    assert snap["default"]["keep_rate"] is None


def test_selected_count_increments_on_set():
    set_active("warm")
    set_active("playful")
    set_active("warm")
    snap = usage_snapshot()
    assert snap["warm"]["selected"] == 2
    assert snap["playful"]["selected"] == 1


# ---- persistence helper ------------------------------------------------

class _FakeStore:
    def __init__(self):
        self.facts = {}

    def find_facts(self, *, kind, key):
        v = self.facts.get((kind, key))
        if v is None:
            return []
        class F:
            value = v
        return [F()]

    def write_fact(self, *, kind, key, value):
        self.facts[(kind, key)] = value


class _FakeMemory:
    def __init__(self):
        self._store = _FakeStore()

    def write_preference(self, key, value):
        self._store.write_fact(
            kind="preference", key=key, value=value)


def test_memory_backed_preset_resolves():
    mem = _FakeMemory()
    mem.write_preference("persona_preset", "tutor")
    p = active_preset(memory=mem)
    assert p.name == "tutor"


def test_persist_choice_writes_to_memory():
    from hgr.live_api.persona_voice import persist_choice_to_memory
    mem = _FakeMemory()
    assert persist_choice_to_memory(mem, "jarvis") is True
    facts = mem._store.find_facts(
        kind="preference", key="persona_preset")
    assert facts[0].value == "jarvis"


def test_persist_choice_rejects_unknown():
    from hgr.live_api.persona_voice import persist_choice_to_memory
    mem = _FakeMemory()
    assert persist_choice_to_memory(mem, "nope") is False


# ---- export -----------------------------------------------------------

def test_export_active_for_ui_returns_json():
    import json
    from hgr.live_api.persona_voice import export_active_for_ui
    set_active("jarvis")
    out = export_active_for_ui()
    parsed = json.loads(out)
    assert parsed["name"] == "jarvis"
    assert "display_name" in parsed
    assert "temperature" in parsed
