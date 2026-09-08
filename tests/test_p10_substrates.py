"""Tests for Phase 10 subscription substrates: persona_marketplace,
tts_voice, privacy_tier, federation."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import hgr.live_api.persona_voice as pv  # noqa: E402
import hgr.live_api.persona_marketplace as pm  # noqa: E402
import hgr.live_api.tts_voice as tts  # noqa: E402
import hgr.live_api.privacy_tier as pt  # noqa: E402
from hgr.live_api.federation import (  # noqa: E402
    FederatedArtifact, FederationEngine, InMemoryRemote,
    SyncReport, reset_global as fed_reset,
)


def setup_function():
    pv.reset_for_tests()
    tts.reset_for_tests()
    pt.reset_for_tests()
    fed_reset()


# ============ persona_marketplace ===============================

def test_catalogue_returns_cards():
    cards = pm.catalogue()
    assert len(cards) >= 4
    names = {c.name for c in cards}
    assert "default" in names
    assert "jarvis" in names


def test_card_for_returns_match():
    card = pm.card_for("jarvis")
    assert card is not None
    assert card.tier == "pro"


def test_card_for_case_insensitive():
    assert pm.card_for("JARVIS") is not None


def test_card_for_missing_returns_none():
    assert pm.card_for("nonexistent") is None


def test_card_sample_reply_populated():
    card = pm.card_for("jarvis")
    assert card.sample_reply


def test_card_to_dict_serializable():
    card = pm.card_for("default")
    d = card.to_dict()
    assert d["name"] == "default"
    assert "display_name" in d
    assert "tier" in d


def test_is_pro_only_correct():
    assert pm.is_pro_only("jarvis") is True
    assert pm.is_pro_only("default") is False


def test_active_card_reflects_active():
    pv.set_active("warm")
    active = pm.active_card()
    assert active is not None
    assert active.name == "warm"


def test_activate_switches_active():
    assert pm.activate("playful") is True
    assert pv.active_preset().name == "playful"


def test_activate_rejects_unknown():
    assert pm.activate("nope") is False


def test_activate_pro_when_disallowed():
    assert pm.activate("jarvis", allow_pro=False) is False


def test_activate_pro_when_allowed():
    assert pm.activate("jarvis", allow_pro=True) is True


def test_catalogue_orders_free_first():
    cards = pm.catalogue()
    tiers_in_order = [c.tier for c in cards]
    # Find first 'pro' index — all preceding must be 'free'.
    first_pro = next((i for i, t in enumerate(tiers_in_order)
                      if t == "pro"), len(tiers_in_order))
    assert all(t == "free"
               for t in tiers_in_order[:first_pro])


# ============ tts_voice ============================================

def test_tts_catalogue_non_empty():
    cat = tts.catalogue()
    assert len(cat) >= 2


def test_tts_get_voice_known():
    v = tts.get_voice("sapi:zira")
    assert v is not None
    assert v.provider == tts.TTSProvider.SAPI


def test_tts_get_voice_unknown():
    assert tts.get_voice("nope:never") is None


def test_tts_get_voice_blank():
    assert tts.get_voice("") is None


def test_tts_set_active_sticks():
    assert tts.set_active("sapi:david") is True
    assert tts.active_voice().voice_id == "sapi:david"


def test_tts_set_active_rejects_unknown():
    assert tts.set_active("does:not:exist") is False


def test_tts_active_falls_back_to_sapi():
    # No active set → returns first SAPI voice.
    v = tts.active_voice()
    assert v.provider == tts.TTSProvider.SAPI


def test_tts_privacy_tier_blocks_cloud():
    tts.set_active("eleven:rachel")
    pt.set_tier(pt.PrivacyTier.LOCAL_ONLY)
    v = tts.active_voice()
    # Cloud voice gets demoted to SAPI default.
    assert v.provider == tts.TTSProvider.SAPI


def test_tts_register_user_clone():
    voice = tts.register_user_clone(
        voice_id="user:konstantin",
        display_name="Konstantin (cloned)")
    assert voice.provider == tts.TTSProvider.USER_CLONE
    assert tts.get_voice("user:konstantin") is not None


def test_tts_register_user_clone_validates_inputs():
    import pytest
    with pytest.raises(ValueError):
        tts.register_user_clone(voice_id="",
                                 display_name="x")


def test_tts_voice_to_dict():
    v = tts.get_voice("sapi:zira")
    d = v.to_dict()
    assert d["voice_id"] == "sapi:zira"
    assert d["provider"] == "sapi"


# ============ privacy_tier ========================================

def test_privacy_tier_defaults_to_cloud():
    assert pt.get_tier() == pt.PrivacyTier.CLOUD


def test_privacy_tier_set_via_enum():
    assert pt.set_tier(pt.PrivacyTier.LOCAL_ONLY) is True
    assert pt.is_local_only() is True


def test_privacy_tier_set_via_string():
    assert pt.set_tier("local_only") is True
    assert pt.is_local_only() is True


def test_privacy_tier_set_invalid_string_rejected():
    assert pt.set_tier("bogus") is False
    assert pt.get_tier() == pt.PrivacyTier.CLOUD


def test_privacy_tier_is_cloud_helper():
    pt.set_tier(pt.PrivacyTier.CLOUD)
    assert pt.is_cloud() is True
    assert pt.is_local_only() is False


def test_privacy_tier_env_var_resolves():
    import os
    from unittest.mock import patch
    with patch.dict(os.environ,
                    {"TOUCHLESS_PRIVACY_TIER": "local_only"}):
        # Reset override so env wins.
        pt.reset_for_tests()
        assert pt.is_local_only() is True


def test_privacy_tier_headline_label():
    assert "Cloud" in pt.headline_label(pt.PrivacyTier.CLOUD)
    assert "Local" in pt.headline_label(
        pt.PrivacyTier.LOCAL_ONLY)


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


def test_privacy_tier_memory_backed():
    mem = _FakeMemory()
    mem.write_preference("privacy_tier", "local_only")
    pt.reset_for_tests()
    assert pt.is_local_only(memory=mem) is True


def test_privacy_tier_persist_writes():
    mem = _FakeMemory()
    assert pt.persist_choice(
        mem, pt.PrivacyTier.LOCAL_ONLY) is True
    facts = mem._store.find_facts(
        kind="preference", key="privacy_tier")
    assert facts[0].value == "local_only"


# ============ federation ==========================================

class _FakeProvider:
    def __init__(self, exports=None):
        self._exports = list(exports or [])
        self.applied: list = []

    def export(self, since):
        out = [a for a in self._exports
               if a.updated_at > since]
        return out

    def apply(self, artifact):
        self.applied.append(artifact)
        return True


def test_artifact_serialize_round_trip():
    a = FederatedArtifact(
        artifact_id="x1", kind="memory",
        device_id="dev-a", version=1,
        updated_at=1.0, payload={"k": "v"})
    blob = a.serialize()
    a2 = FederatedArtifact.deserialize(blob)
    assert a2.artifact_id == "x1"
    assert a2.payload == {"k": "v"}


def test_in_memory_remote_push_then_pull():
    r = InMemoryRemote()
    a = FederatedArtifact(
        artifact_id="x1", kind="memory",
        device_id="dev-b", version=5,
        updated_at=1.0, payload={"k": "v"})
    r.push([a])
    pulled = r.pull({"dev-b": 0})
    assert len(pulled) == 1


def test_in_memory_remote_filters_by_version():
    r = InMemoryRemote()
    r.push([FederatedArtifact(
        artifact_id="x1", kind="memory",
        device_id="dev-b", version=3,
        updated_at=1.0, payload={})])
    r.push([FederatedArtifact(
        artifact_id="x2", kind="memory",
        device_id="dev-b", version=4,
        updated_at=1.0, payload={})])
    pulled = r.pull({"dev-b": 3})
    assert len(pulled) == 1
    assert pulled[0].artifact_id == "x2"


def test_in_memory_remote_replaces_same_id():
    r = InMemoryRemote()
    r.push([FederatedArtifact(
        artifact_id="x1", kind="memory",
        device_id="dev-b", version=1,
        updated_at=1.0, payload={"v": "old"})])
    r.push([FederatedArtifact(
        artifact_id="x1", kind="memory",
        device_id="dev-b", version=2,
        updated_at=2.0, payload={"v": "new"})])
    all_a = r.all()
    assert len(all_a) == 1
    assert all_a[0].payload == {"v": "new"}


def test_engine_no_remote_returns_empty_report():
    e = FederationEngine()
    rep = e.sync_once()
    assert rep.pushed == 0
    assert rep.pulled == 0


def test_engine_push_and_pull():
    r = InMemoryRemote()
    prov = _FakeProvider(exports=[
        FederatedArtifact(
            artifact_id="a1", kind="memory",
            device_id="dev-a", version=1,
            updated_at=2.0, payload={}),
    ])
    e = FederationEngine(device_id="dev-a", remote=r)
    e.register_provider("memory", prov)
    rep = e.sync_once()
    assert rep.pushed == 1
    assert len(r.all()) == 1


def test_engine_applies_pulled_artifacts():
    r = InMemoryRemote()
    r.push([FederatedArtifact(
        artifact_id="a1", kind="memory",
        device_id="other-dev", version=1,
        updated_at=1.0, payload={})])
    prov = _FakeProvider()
    e = FederationEngine(device_id="dev-a", remote=r)
    e.register_provider("memory", prov)
    rep = e.sync_once()
    assert rep.applied == 1
    assert len(prov.applied) == 1


def test_engine_skips_unknown_kinds():
    r = InMemoryRemote()
    r.push([FederatedArtifact(
        artifact_id="a1", kind="unknown",
        device_id="other-dev", version=1,
        updated_at=1.0, payload={})])
    e = FederationEngine(device_id="dev-a", remote=r)
    rep = e.sync_once()
    # Pulled but not applied — no provider for 'unknown'.
    assert rep.applied == 0


def test_engine_reset_state_clears_watermarks():
    r = InMemoryRemote()
    e = FederationEngine(device_id="dev-a", remote=r)
    e._pulled_versions["x"] = 99
    e.reset_state()
    assert e._pulled_versions == {}


def test_engine_set_remote_None_disables():
    r = InMemoryRemote()
    e = FederationEngine(device_id="dev-a", remote=r)
    e.set_remote(None)
    rep = e.sync_once()
    assert rep.pushed == 0
    assert rep.pulled == 0


def test_engine_device_id_assigned():
    e = FederationEngine(device_id="my-dev")
    assert e.device_id == "my-dev"


def test_engine_handles_provider_exception_in_export():
    class BadProv:
        def export(self, since):
            raise RuntimeError("nope")
        def apply(self, artifact):
            return True
    e = FederationEngine(
        device_id="dev-a", remote=InMemoryRemote())
    e.register_provider("memory", BadProv())
    rep = e.sync_once()
    # Failure recorded; doesn't crash.
    assert rep.failed >= 1


def test_engine_handles_provider_exception_in_apply():
    r = InMemoryRemote()
    r.push([FederatedArtifact(
        artifact_id="a1", kind="memory",
        device_id="other-dev", version=1,
        updated_at=1.0, payload={})])
    class BadProv:
        def export(self, since):
            return []
        def apply(self, artifact):
            raise RuntimeError("nope")
    e = FederationEngine(device_id="dev-a", remote=r)
    e.register_provider("memory", BadProv())
    rep = e.sync_once()
    assert rep.failed >= 1


def test_sync_report_default_zero():
    r = SyncReport()
    assert r.pushed == 0
    assert r.pulled == 0
    assert r.duration_ms == 0.0
