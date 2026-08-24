"""Persona marketplace — UI descriptors for the voice catalogue.

Phase-10 subscription polish. The `persona_voice` module already
owns the named preset catalogue + activation. The marketplace is
the UI-facing wrapper: each entry rendered as a card with display
name, description, sample reply, and a one-click activation hook.

Cards are sourced from:
  1. Built-in presets in `persona_voice` (always available).
  2. Subscription-tier extras (Jarvis-tribute Pro, voice clones, …)
     — surfaced when the subscriber's tier supports them.
  3. User-saved custom presets (created via "save current as …").

The marketplace is consumed by the chat panel's persona-picker
strip. The data shape is stable so UI can render cards without
needing direct access to `persona_voice` internals.

Author: Konstantin Markov
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass(frozen=True)
class MarketplaceCard:
    name: str                       # persona preset slug
    display_name: str
    description: str
    tier: str = "free"              # "free" | "pro" | "team"
    sample_reply: str = ""
    accent_color: str = ""
    badge: str = ""                 # e.g. "PRO" / "NEW"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "display_name": self.display_name,
            "description": self.description,
            "tier": self.tier,
            "sample_reply": self.sample_reply,
            "accent_color": self.accent_color,
            "badge": self.badge,
        }


# Stable sample replies surfaced in the card preview. Pulled from
# each preset's first few-shot example so they actually demo the
# voice. Falls back to "" when the preset has no examples.
def _first_example_reply(preset_name: str) -> str:
    try:
        from . import persona_voice
        p = persona_voice.get_preset(preset_name)
        if p is None or not p.examples:
            return ""
        return p.examples[0][1]
    except Exception:
        return ""


_TIER_BY_NAME: Dict[str, str] = {
    "default": "free",
    "concise": "free",
    "warm": "free",
    "playful": "free",
    "tutor": "free",
    "jarvis": "pro",      # the dry-Jarvis voice is a Pro perk
}


_ACCENT_BY_NAME: Dict[str, str] = {
    "default": "#7AB7FF",
    "jarvis":  "#D4A24A",
    "concise": "#888888",
    "warm":    "#FFB084",
    "playful": "#C589FF",
    "tutor":   "#6FDA7C",
}


def _build_card(preset_name: str) -> Optional[MarketplaceCard]:
    try:
        from . import persona_voice
        preset = persona_voice.get_preset(preset_name)
        if preset is None:
            return None
        return MarketplaceCard(
            name=preset.name,
            display_name=preset.display_name,
            description=preset.description,
            tier=_TIER_BY_NAME.get(preset.name, "free"),
            sample_reply=_first_example_reply(preset.name),
            accent_color=_ACCENT_BY_NAME.get(preset.name, ""),
            badge=("PRO"
                   if _TIER_BY_NAME.get(preset.name) == "pro"
                   else ""),
        )
    except Exception:
        return None


def catalogue() -> List[MarketplaceCard]:
    """Return all available marketplace cards. Order: free first
    (alphabetical), then pro (alphabetical)."""
    try:
        from . import persona_voice
        presets = persona_voice.all_presets()
    except Exception:
        return []
    cards: List[MarketplaceCard] = []
    for p in presets:
        card = _build_card(p.name)
        if card is not None:
            cards.append(card)
    cards.sort(
        key=lambda c: (0 if c.tier == "free" else 1,
                       c.display_name.lower()))
    return cards


def card_for(name: str) -> Optional[MarketplaceCard]:
    if not name:
        return None
    return _build_card(name.strip().lower())


def is_pro_only(name: str) -> bool:
    return _TIER_BY_NAME.get((name or "").lower(), "free") == "pro"


def active_card() -> Optional[MarketplaceCard]:
    try:
        from . import persona_voice
        p = persona_voice.active_preset()
        return _build_card(p.name)
    except Exception:
        return None


def activate(name: str, *, memory: Optional[Any] = None,
             allow_pro: bool = True) -> bool:
    """Activate a preset by slug. Returns False when:
      * Unknown slug
      * Pro preset selected by a free user (allow_pro=False)"""
    if not name:
        return False
    slug = name.strip().lower()
    if is_pro_only(slug) and not allow_pro:
        return False
    try:
        from . import persona_voice
        ok = persona_voice.set_active(slug)
        if ok and memory is not None:
            persona_voice.persist_choice_to_memory(memory, slug)
        return ok
    except Exception:
        return False
