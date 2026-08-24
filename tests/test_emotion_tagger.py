"""Tests for emotion_tagger (Phase 7 polish: JARVIS-grade tone)."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from hgr.live_api.emotion_tagger import (  # noqa: E402
    Emotion, EmotionTag, _classify_heuristic, _snap_to_persona,
    classify, combine_instructions, delivery_hint, reset_cache,
)


def setup_function():
    reset_cache()
    # Avoid LLM calls during tests by removing the API key.
    os.environ.pop("OPENAI_API_KEY", None)


# ---- heuristic --------------------------------------------------------

def test_heuristic_apologetic_on_couldnt():
    tag = _classify_heuristic("Couldn't queue that — Spotify "
                               "wasn't reachable.")
    assert tag.emotion == Emotion.APOLOGETIC


def test_heuristic_apologetic_on_failed():
    tag = _classify_heuristic("Build failed at step 3.")
    assert tag.emotion == Emotion.APOLOGETIC


def test_heuristic_urgent_on_critical():
    tag = _classify_heuristic("CRITICAL: deadline in 5 minutes.")
    assert tag.emotion == Emotion.URGENT


def test_heuristic_happy_on_done():
    tag = _classify_heuristic("Done — all green.")
    assert tag.emotion == Emotion.HAPPY


def test_heuristic_excited_on_double_bang():
    tag = _classify_heuristic("Finally working!!")
    assert tag.emotion == Emotion.EXCITED


def test_heuristic_playful_on_em_dashes():
    tag = _classify_heuristic("Cueing it up — and yes, again — enjoy.")
    assert tag.emotion == Emotion.PLAYFUL


def test_heuristic_curious_on_question_mark():
    tag = _classify_heuristic("Want me to dig deeper?")
    assert tag.emotion == Emotion.CURIOUS


def test_heuristic_dry_on_naturally():
    tag = _classify_heuristic("Naturally, the build broke.")
    assert tag.emotion == Emotion.DRY


def test_heuristic_neutral_default():
    tag = _classify_heuristic("Today is Thursday.")
    assert tag.emotion == Emotion.NEUTRAL


def test_heuristic_empty_returns_neutral():
    tag = _classify_heuristic("")
    assert tag.emotion == Emotion.NEUTRAL


# ---- persona snapping ------------------------------------------------

def test_snap_jarvis_playful_becomes_dry():
    assert (_snap_to_persona(Emotion.PLAYFUL, "jarvis")
            == Emotion.DRY)


def test_snap_jarvis_happy_becomes_dry():
    assert (_snap_to_persona(Emotion.HAPPY, "jarvis")
            == Emotion.DRY)


def test_snap_jarvis_neutral_stays():
    assert (_snap_to_persona(Emotion.NEUTRAL, "jarvis")
            == Emotion.NEUTRAL)


def test_snap_jarvis_apologetic_stays():
    assert (_snap_to_persona(Emotion.APOLOGETIC, "jarvis")
            == Emotion.APOLOGETIC)


def test_snap_concise_excited_becomes_neutral():
    assert (_snap_to_persona(Emotion.EXCITED, "concise")
            == Emotion.NEUTRAL)


def test_snap_concise_urgent_stays():
    assert (_snap_to_persona(Emotion.URGENT, "concise")
            == Emotion.URGENT)


def test_snap_playful_concerned_softens_to_neutral():
    assert (_snap_to_persona(Emotion.CONCERNED, "playful")
            == Emotion.NEUTRAL)


def test_snap_playful_excited_stays():
    assert (_snap_to_persona(Emotion.EXCITED, "playful")
            == Emotion.EXCITED)


def test_snap_warm_dry_becomes_neutral():
    assert (_snap_to_persona(Emotion.DRY, "warm")
            == Emotion.NEUTRAL)


def test_snap_warm_playful_becomes_happy():
    assert (_snap_to_persona(Emotion.PLAYFUL, "warm")
            == Emotion.HAPPY)


def test_snap_tutor_collapses_to_thoughtful():
    """Tutor's main delivery is thoughtful — most emotions snap."""
    assert (_snap_to_persona(Emotion.EXCITED, "tutor")
            == Emotion.THOUGHTFUL)


def test_snap_default_passes_through():
    assert (_snap_to_persona(Emotion.PLAYFUL, "default")
            == Emotion.PLAYFUL)


# ---- delivery hints --------------------------------------------------

def test_delivery_hint_returns_non_empty_for_every_emotion():
    for e in Emotion:
        assert delivery_hint(e)
        assert len(delivery_hint(e)) > 20


def test_delivery_hint_dry_mentions_deadpan():
    assert "deadpan" in delivery_hint(Emotion.DRY).lower()


def test_delivery_hint_apologetic_mentions_softer():
    h = delivery_hint(Emotion.APOLOGETIC).lower()
    assert "soft" in h or "gentl" in h


def test_delivery_hint_urgent_mentions_alert():
    h = delivery_hint(Emotion.URGENT).lower()
    assert "alert" in h or "crisp" in h or "sharp" in h


# ---- combine_instructions --------------------------------------------

def test_combine_keeps_base_and_appends_hint():
    base = "Read warmly."
    out = combine_instructions(base, Emotion.HAPPY)
    assert "Read warmly" in out
    assert delivery_hint(Emotion.HAPPY) in out


def test_combine_handles_empty_base():
    out = combine_instructions("", Emotion.HAPPY)
    assert out == delivery_hint(Emotion.HAPPY)


def test_combine_handles_none_base():
    out = combine_instructions(None, Emotion.NEUTRAL)
    assert out == delivery_hint(Emotion.NEUTRAL)


# ---- top-level classify (heuristic path, no API key) -----------------

def test_classify_no_api_key_uses_heuristic():
    tag = classify("Couldn't reach Spotify.")
    assert tag.emotion == Emotion.APOLOGETIC
    assert "heuristic" in tag.source


def test_classify_empty_text_neutral():
    tag = classify("")
    assert tag.emotion == Emotion.NEUTRAL


def test_classify_uses_persona_snap():
    """Jarvis + playful content → DRY (snapped)."""
    tag = classify("Cueing it up — enjoy.", persona="jarvis")
    assert tag.emotion == Emotion.DRY


def test_classify_concise_with_excited_content():
    tag = classify("Finally working!!", persona="concise")
    assert tag.emotion in (Emotion.NEUTRAL, Emotion.URGENT)


def test_classify_cache_returns_same_tag():
    tag1 = classify("Build failed at step 3.")
    tag2 = classify("Build failed at step 3.")
    assert tag1.emotion == tag2.emotion


def test_classify_different_persona_different_cache_key():
    tag_a = classify("Cueing it up — enjoy.", persona="default")
    tag_b = classify("Cueing it up — enjoy.", persona="jarvis")
    # Different persona → different snap result.
    assert tag_a.emotion != tag_b.emotion


# ---- LLM path (mocked) -----------------------------------------------

def _mock_llm_response(emotion: str, confidence: float = 0.9,
                       reason: str = "ok"):
    body = {
        "choices": [{
            "message": {
                "content": json.dumps({
                    "emotion": emotion,
                    "confidence": confidence,
                    "reason": reason,
                })}}]
    }
    class _Resp:
        status = 200
        def __init__(self, b): self._b = b
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self):
            return json.dumps(self._b).encode("utf-8")
    return _Resp(body)


def test_classify_llm_path_when_key_present():
    with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}):
        with patch("urllib.request.urlopen",
                    return_value=_mock_llm_response("happy", 0.95)):
            tag = classify("Build's green for once.")
    assert tag.emotion == Emotion.HAPPY
    assert "llm" in tag.source


def test_classify_llm_failure_falls_back_to_heuristic():
    import urllib.error
    with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}):
        with patch("urllib.request.urlopen",
                    side_effect=urllib.error.HTTPError(
                        "u", 500, "x", {}, None)):
            tag = classify("Couldn't reach Spotify.")
    # Heuristic catches "couldn't"
    assert tag.emotion == Emotion.APOLOGETIC


def test_classify_llm_invalid_emotion_falls_back():
    with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}):
        with patch("urllib.request.urlopen",
                    return_value=_mock_llm_response("ecstatic", 1.0)):
            tag = classify("Done — all green.")
    # Invalid emotion → fallback to heuristic, which sees "done +
    # all green" → HAPPY.
    assert tag.emotion == Emotion.HAPPY


def test_classify_llm_with_persona_snap():
    """LLM returns playful, persona is jarvis → snapped to DRY."""
    with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}):
        with patch("urllib.request.urlopen",
                    return_value=_mock_llm_response("playful", 0.9)):
            tag = classify("Made you look.", persona="jarvis")
    assert tag.emotion == Emotion.DRY
    assert "snap" in tag.source


# ---- EmotionTag dataclass --------------------------------------------

def test_emotion_tag_defaults():
    t = EmotionTag(Emotion.NEUTRAL)
    assert t.confidence == 0.0
    assert t.source == ""
