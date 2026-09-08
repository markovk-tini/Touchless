"""Tool discovery loop — when Iris can't do it, suggest a way to
enable it.

Phase-9 cognition. Today when the user asks for something outside
Iris's tool catalogue, the reply is "I can't do that." For most
common gaps (Notion, Linear, Jira, Obsidian, Figma, GitHub, etc.)
there's an MCP server that bridges it — Iris just doesn't know
about them unless they're already installed.

This module owns the catalogue + the matcher. When `try_handle`
falls through to "no tool matches", we run:

  * `suggest_for_text(text)` — scans for capability keywords, returns
    a list of `ToolSuggestion` with the MCP server's npm/pip package,
    a one-line description, and a confidence score.

The orchestrator can then surface "I can read Notion if you install
the Notion MCP server — want me to set that up? (one click)."

Catalogue is bundled and stable; adding new servers is a code
edit. The catalogue is INTENTIONALLY narrow:
  * Only widely-used servers
  * Only those with stated free-tier or open-source backing
  * Only those where the OAuth / install flow is well-defined

Each suggestion includes a one-line `install_hint` the UI can
turn into a button.

Author: Konstantin Markov
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass(frozen=True)
class ToolSuggestion:
    capability: str                  # human label ("Notion")
    keywords: Tuple[str, ...]        # for matching
    mcp_server: str                  # npm package or git URL
    description: str = ""
    install_hint: str = ""
    auth_kind: str = "oauth"         # oauth | api_key | none
    confidence: float = 0.7

    def install_button_label(self) -> str:
        return f"Enable {self.capability}"

    def to_dict(self) -> Dict[str, object]:
        return {
            "capability": self.capability,
            "keywords": list(self.keywords),
            "mcp_server": self.mcp_server,
            "description": self.description,
            "install_hint": self.install_hint,
            "auth_kind": self.auth_kind,
            "confidence": self.confidence,
        }


# Note: each suggestion's keywords are matched as whole words.
# Order matters when multiple capabilities share keywords —
# more-specific ones go first.
_CATALOGUE: Tuple[ToolSuggestion, ...] = (
    ToolSuggestion(
        capability="Notion",
        keywords=("notion",),
        mcp_server="@notionhq/notion-mcp-server",
        description=(
            "Read + write Notion pages, databases, and blocks."),
        install_hint=(
            "Run 'npx @notionhq/notion-mcp-server' and "
            "complete the OAuth handshake."),
        auth_kind="oauth"),
    ToolSuggestion(
        capability="Linear",
        keywords=("linear", "tickets", "linear ticket"),
        mcp_server="@linear/linear-mcp",
        description=(
            "Query and update Linear issues + cycles + projects."),
        install_hint=(
            "Sign in with Linear OAuth via the MCP picker."),
        auth_kind="oauth"),
    ToolSuggestion(
        capability="Jira",
        keywords=("jira", "atlassian"),
        mcp_server="jira-mcp",
        description=(
            "Query + comment on Jira tickets."),
        install_hint=(
            "Paste your Atlassian site URL + API token in the "
            "MCP picker."),
        auth_kind="api_key"),
    ToolSuggestion(
        capability="Obsidian",
        keywords=("obsidian", "vault"),
        mcp_server="mcp-obsidian",
        description=(
            "Read and search your local Obsidian vault."),
        install_hint=(
            "Point the MCP picker at your vault folder. "
            "No network access needed."),
        auth_kind="none"),
    ToolSuggestion(
        capability="Figma",
        keywords=("figma", "design file", "mockup"),
        mcp_server="figma-mcp",
        description=(
            "Read Figma files, frames, and design tokens."),
        install_hint=(
            "Generate a Figma personal access token and paste it "
            "into the MCP picker."),
        auth_kind="api_key"),
    ToolSuggestion(
        capability="GitHub",
        keywords=("github", "github issue", "pr"),
        mcp_server="@modelcontextprotocol/server-github",
        description=(
            "Read PRs / issues / code from your GitHub repos."),
        install_hint=(
            "Run 'npx @modelcontextprotocol/server-github' "
            "and sign in with GitHub OAuth."),
        auth_kind="oauth"),
    ToolSuggestion(
        capability="Slack",
        keywords=("slack", "slack channel", "slack dm"),
        mcp_server="@modelcontextprotocol/server-slack",
        description=(
            "Read + post in Slack channels you can access."),
        install_hint=(
            "Sign in with Slack OAuth via the MCP picker."),
        auth_kind="oauth"),
    ToolSuggestion(
        capability="Discord",
        keywords=("discord", "discord message"),
        mcp_server="@modelcontextprotocol/server-discord",
        description=(
            "Read + send Discord messages in your servers."),
        install_hint=(
            "Create a Discord bot token + invite it to your "
            "server. Paste the token in the MCP picker."),
        auth_kind="api_key"),
    ToolSuggestion(
        capability="Spotify",
        keywords=("spotify",),
        mcp_server="spotify-mcp",
        description=(
            "Search + queue tracks, control playback."),
        install_hint=(
            "Sign in with Spotify OAuth via the MCP picker."),
        auth_kind="oauth"),
)


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").lower().strip())


def _has_keyword(text: str, kw: str) -> bool:
    """Whole-word match of `kw` in `text`. Multi-word keywords
    match as substrings (e.g. 'linear ticket' inside 'create a
    linear ticket')."""
    if " " in kw:
        return kw in text
    return bool(re.search(rf"\b{re.escape(kw)}\b", text))


def suggest_for_text(text: str,
                     *, min_confidence: float = 0.0
                     ) -> List[ToolSuggestion]:
    """Return the catalogue suggestions whose keywords appear in
    `text`. Ordered by descending confidence."""
    norm = _normalize(text)
    if not norm:
        return []
    hits: List[ToolSuggestion] = []
    seen: set = set()
    for sug in _CATALOGUE:
        if sug.capability in seen:
            continue
        if any(_has_keyword(norm, kw) for kw in sug.keywords):
            if sug.confidence >= min_confidence:
                hits.append(sug)
                seen.add(sug.capability)
    hits.sort(key=lambda s: s.confidence, reverse=True)
    return hits


def get_by_capability(capability: str) -> Optional[ToolSuggestion]:
    """Lookup by capability label (case-insensitive)."""
    if not capability:
        return None
    cap = capability.lower()
    for sug in _CATALOGUE:
        if sug.capability.lower() == cap:
            return sug
    return None


def all_suggestions() -> List[ToolSuggestion]:
    return list(_CATALOGUE)


def already_installed(suggestion: ToolSuggestion,
                      *, mcp_registry: Optional[object] = None
                      ) -> bool:
    """Best-effort check: is the MCP server already registered?
    Returns True only when the registry confirms; False on any
    error or missing registry. Conservative — false positives
    would suppress a useful suggestion."""
    if mcp_registry is None:
        try:
            from .mcp_registry import global_registry
            mcp_registry = global_registry()
        except Exception:
            return False
    try:
        servers = list(getattr(mcp_registry, "servers", []))
        target = suggestion.mcp_server.lower()
        for s in servers:
            name = (getattr(s, "name", "")
                    or getattr(s, "package", "")
                    or str(s)).lower()
            if target in name or name in target:
                return True
    except Exception:
        return False
    return False


def render_offer_text(sug: ToolSuggestion) -> str:
    """One-line user-facing 'want to enable this?' offer."""
    return (f"I can do that with the {sug.capability} MCP server — "
            f"{sug.description} Want to set it up? "
            f"({sug.install_hint})")
