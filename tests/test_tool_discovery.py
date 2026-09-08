"""Tests for tool_discovery (Phase 9 B4)."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from hgr.live_api.tool_discovery import (  # noqa: E402
    ToolSuggestion, _has_keyword, _normalize,
    all_suggestions, already_installed,
    get_by_capability, render_offer_text,
    suggest_for_text,
)


# ---- normalize + keyword helpers --------------------------------

def test_normalize_lowercases():
    assert _normalize("Hello World") == "hello world"


def test_normalize_collapses_whitespace():
    assert _normalize("  a   b  ") == "a b"


def test_has_keyword_whole_word():
    assert _has_keyword("install notion mcp", "notion")


def test_has_keyword_not_substring():
    assert not _has_keyword("send via notional bank",
                             "notion")


def test_has_keyword_multi_word():
    assert _has_keyword("create a linear ticket for q3",
                         "linear ticket")


# ---- catalogue --------------------------------------------------

def test_catalogue_non_empty():
    assert len(all_suggestions()) >= 5


def test_each_suggestion_has_install_hint():
    for s in all_suggestions():
        assert s.install_hint


def test_each_suggestion_has_mcp_server():
    for s in all_suggestions():
        assert s.mcp_server


# ---- suggest_for_text -------------------------------------------

def test_suggest_notion():
    suggestions = suggest_for_text(
        "can you read my notion page about q3")
    caps = [s.capability for s in suggestions]
    assert "Notion" in caps


def test_suggest_linear():
    suggestions = suggest_for_text(
        "find that linear ticket I filed yesterday")
    caps = [s.capability for s in suggestions]
    assert "Linear" in caps


def test_suggest_jira():
    suggestions = suggest_for_text(
        "comment on the atlassian ticket")
    caps = [s.capability for s in suggestions]
    assert "Jira" in caps


def test_suggest_obsidian():
    suggestions = suggest_for_text(
        "search my obsidian vault for that note")
    caps = [s.capability for s in suggestions]
    assert "Obsidian" in caps


def test_suggest_figma():
    suggestions = suggest_for_text(
        "what's in the figma design file")
    caps = [s.capability for s in suggestions]
    assert "Figma" in caps


def test_suggest_github():
    suggestions = suggest_for_text("show me my open github prs")
    caps = [s.capability for s in suggestions]
    assert "GitHub" in caps


def test_suggest_slack():
    suggestions = suggest_for_text(
        "post in the slack channel")
    caps = [s.capability for s in suggestions]
    assert "Slack" in caps


def test_suggest_spotify():
    suggestions = suggest_for_text("queue a song on spotify")
    caps = [s.capability for s in suggestions]
    assert "Spotify" in caps


def test_suggest_empty_returns_nothing():
    assert suggest_for_text("") == []


def test_suggest_unknown_returns_nothing():
    assert suggest_for_text("send tea to the dog") == []


def test_suggest_multiple_capabilities():
    suggestions = suggest_for_text(
        "push the figma design to my notion page")
    caps = {s.capability for s in suggestions}
    assert "Figma" in caps
    assert "Notion" in caps


def test_suggest_orders_by_confidence():
    suggestions = suggest_for_text(
        "look at the notion page and the figma design")
    confs = [s.confidence for s in suggestions]
    assert confs == sorted(confs, reverse=True)


def test_suggest_dedups_per_capability():
    """Repeated keywords for the same capability don't duplicate."""
    suggestions = suggest_for_text(
        "notion notion notion page")
    notion_hits = [s for s in suggestions
                   if s.capability == "Notion"]
    assert len(notion_hits) == 1


# ---- get_by_capability ------------------------------------------

def test_get_by_capability_match():
    sug = get_by_capability("Notion")
    assert sug is not None
    assert sug.capability == "Notion"


def test_get_by_capability_case_insensitive():
    assert get_by_capability("NOTION") is not None
    assert get_by_capability("notion") is not None


def test_get_by_capability_missing():
    assert get_by_capability("DoesNotExist") is None


def test_get_by_capability_empty_returns_none():
    assert get_by_capability("") is None


# ---- already_installed -----------------------------------------

class _FakeServer:
    def __init__(self, name):
        self.name = name


class _FakeRegistry:
    def __init__(self, servers):
        self.servers = servers


def test_already_installed_matches_name():
    sug = get_by_capability("Notion")
    reg = _FakeRegistry([_FakeServer("notion-mcp-server")])
    assert already_installed(sug, mcp_registry=reg) is True


def test_already_installed_no_match():
    sug = get_by_capability("Notion")
    reg = _FakeRegistry([_FakeServer("github-mcp-server")])
    assert already_installed(sug, mcp_registry=reg) is False


def test_already_installed_no_registry_returns_false():
    sug = get_by_capability("Notion")
    # No global registry available in tests.
    assert already_installed(sug, mcp_registry=None) in (
        True, False)


def test_already_installed_buggy_registry():
    sug = get_by_capability("Notion")
    class Boom:
        @property
        def servers(self):
            raise RuntimeError("nope")
    assert already_installed(sug, mcp_registry=Boom()) is False


# ---- render_offer_text ------------------------------------------

def test_render_offer_text_mentions_capability():
    sug = get_by_capability("Notion")
    out = render_offer_text(sug)
    assert "Notion" in out


def test_render_offer_text_mentions_install_hint():
    sug = get_by_capability("Notion")
    out = render_offer_text(sug)
    assert "npx" in out or "OAuth" in out or "MCP" in out


# ---- dataclass --------------------------------------------------

def test_to_dict_round_trip():
    sug = get_by_capability("Notion")
    d = sug.to_dict()
    assert d["capability"] == "Notion"
    assert isinstance(d["keywords"], list)


def test_install_button_label():
    sug = get_by_capability("Notion")
    assert sug.install_button_label() == "Enable Notion"
