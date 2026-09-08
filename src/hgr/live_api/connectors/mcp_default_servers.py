"""Curated catalog of MCP servers Iris knows how to launch.

Each entry tells the UI:
  * how to spawn the server (command + args)
  * what env vars (auth tokens) it needs before it can run
  * a short human-readable summary for the picker
  * a category for grouping
  * the runtime it needs (Node = npx, Python = uvx, etc.) so the UI
    can warn the user if it's missing

The MCP picker reads this list, intersects with what's in
~/Documents/Touchless/mcp_servers.json, and writes user toggles back.
Adding a new server here surfaces it in the picker automatically — no
other code changes needed.

Server IDs and command shapes mirror the public `@modelcontextprotocol`
catalog at https://github.com/modelcontextprotocol/servers (and
mcp.so for third-party ones). Keep one entry per server.

Author: Konstantin Markov
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass(frozen=True)
class McpServerSpec:
    id: str                          # stable id used in mcp_servers.json
    name: str                        # display name in the picker
    category: str                    # "Code", "Productivity", "Search", "Files", "Data", "Web"
    description: str                 # one-line blurb
    runtime: str                     # "node" (npx) or "python" (uvx / pip)
    command: str                     # binary to spawn
    args: List[str] = field(default_factory=list)
    env_required: List[str] = field(default_factory=list)   # must be set to enable
    env_optional: List[str] = field(default_factory=list)   # nice to have
    setup_url: str = ""              # link to where the user gets the token / installs the server
    needs_args: List[str] = field(default_factory=list)     # arg names the user must supply (e.g. filesystem root)

    # Setup friction tier — the user-facing axis that decides what gets
    # auto-enabled, what gets hidden by default, and what shows a setup
    # walkthrough. Independent of `runtime` (a server can be Node-based
    # but tier=zero if Node ships with the installer in the future).
    #
    #   zero     — works the moment Iris starts; no token, no path, no
    #              external binary the user has to install themselves.
    #              Auto-enabled on first launch in the shipped app.
    #   path     — needs the user to supply a path / connection string
    #              (filesystem root, db url). One-time, no account.
    #   oauth    — one-click browser auth flow. We open the consent
    #              page; user clicks Allow; we capture the token.
    #   api_key  — user must paste a token they generated elsewhere.
    #              Power-user / advanced tier; hidden by default in the
    #              shipped app's picker.
    tier: str = "api_key"


# ============================================================================
# THE CATALOG — keep alphabetized within each category for easy scanning.
# ============================================================================

DEFAULT_SERVERS: List[McpServerSpec] = [
    # ---- Code / Dev ------------------------------------------------------
    McpServerSpec(
        id="github",
        name="GitHub",
        category="Code",
        description="Repos, issues, pull requests, code search across your GitHub account.",
        runtime="node",
        command="npx",
        args=["-y", "@modelcontextprotocol/server-github"],
        env_required=["GITHUB_PERSONAL_ACCESS_TOKEN"],
        setup_url="https://github.com/settings/tokens",
        tier="api_key",
    ),
    McpServerSpec(
        id="git",
        name="Git",
        category="Code",
        description="Local git operations on a repo path you specify (log, diff, blame, branches).",
        runtime="python",
        command="uvx",
        args=["mcp-server-git"],
        setup_url="https://github.com/modelcontextprotocol/servers/tree/main/src/git",
        tier="zero",  # works the moment uvx is on PATH (we bundle that later)
    ),
    McpServerSpec(
        id="sentry",
        name="Sentry",
        category="Code",
        description="Read errors / issues / events from your Sentry project.",
        runtime="python",
        command="uvx",
        args=["mcp-server-sentry"],
        env_required=["SENTRY_AUTH_TOKEN"],
        setup_url="https://sentry.io/settings/account/api/auth-tokens/",
        tier="api_key",
    ),

    # ---- Productivity / Project Management -------------------------------
    McpServerSpec(
        id="slack",
        name="Slack",
        category="Productivity",
        description="Read channels, post messages, search history, list users.",
        runtime="node",
        command="npx",
        args=["-y", "@modelcontextprotocol/server-slack"],
        env_required=["SLACK_BOT_TOKEN", "SLACK_TEAM_ID"],
        setup_url="https://api.slack.com/apps",
        tier="api_key",
    ),
    McpServerSpec(
        id="linear",
        name="Linear",
        category="Productivity",
        description="Issues, projects, cycles, comments in your Linear workspace.",
        runtime="node",
        command="npx",
        args=["-y", "mcp-linear"],
        env_required=["LINEAR_API_KEY"],
        setup_url="https://linear.app/settings/api",
        tier="api_key",
    ),
    McpServerSpec(
        id="notion",
        name="Notion (via MCP)",
        category="Productivity",
        description="Search pages, read databases, append blocks in your Notion workspace.",
        runtime="node",
        command="npx",
        args=["-y", "@notionhq/notion-mcp-server"],
        env_required=["NOTION_API_KEY"],
        setup_url="https://www.notion.so/profile/integrations",
        tier="api_key",
    ),
    McpServerSpec(
        id="atlassian",
        name="Atlassian (Jira + Confluence)",
        category="Productivity",
        description="Jira issues, Confluence pages — read and write.",
        runtime="python",
        command="uvx",
        args=["mcp-atlassian"],
        env_required=["ATLASSIAN_HOST", "ATLASSIAN_EMAIL", "ATLASSIAN_API_TOKEN"],
        setup_url="https://id.atlassian.com/manage-profile/security/api-tokens",
        tier="api_key",
    ),

    # ---- Files / Local ---------------------------------------------------
    McpServerSpec(
        id="filesystem",
        name="Filesystem",
        category="Files",
        description="Read / write files under one or more folders you whitelist. "
                    "Use cautiously — the LLM can edit any file in the listed roots.",
        runtime="node",
        command="npx",
        args=["-y", "@modelcontextprotocol/server-filesystem"],
        needs_args=["root_path"],   # one or more root paths appended to args
        setup_url="https://github.com/modelcontextprotocol/servers/tree/main/src/filesystem",
        tier="path",
    ),
    McpServerSpec(
        id="memory",
        name="Memory (graph)",
        category="Files",
        description="Persistent knowledge graph — entities/relations the model can write to and recall.",
        runtime="node",
        command="npx",
        args=["-y", "@modelcontextprotocol/server-memory"],
        setup_url="https://github.com/modelcontextprotocol/servers/tree/main/src/memory",
        tier="zero",
    ),

    # ---- Search / Web ----------------------------------------------------
    McpServerSpec(
        id="brave_search",
        name="Brave Search",
        category="Search",
        description="Web search via Brave (privacy-preserving, free tier).",
        runtime="node",
        command="npx",
        args=["-y", "@modelcontextprotocol/server-brave-search"],
        env_required=["BRAVE_API_KEY"],
        setup_url="https://brave.com/search/api/",
        tier="api_key",
    ),
    McpServerSpec(
        id="fetch",
        name="Fetch (URL reader)",
        category="Search",
        description="Fetch a URL, extract its main text. Lighter-weight than driving a browser.",
        runtime="python",
        command="uvx",
        args=["mcp-server-fetch"],
        setup_url="https://github.com/modelcontextprotocol/servers/tree/main/src/fetch",
        tier="zero",
    ),
    McpServerSpec(
        id="puppeteer",
        name="Puppeteer (browser)",
        category="Web",
        description="Headless Chrome — navigate, screenshot, click, type. Heavy but powerful.",
        runtime="node",
        command="npx",
        args=["-y", "@modelcontextprotocol/server-puppeteer"],
        setup_url="https://github.com/modelcontextprotocol/servers/tree/main/src/puppeteer",
        tier="zero",  # downloads chromium on first use; no user setup
    ),

    # ---- Data ------------------------------------------------------------
    McpServerSpec(
        id="postgres",
        name="Postgres",
        category="Data",
        description="Read-only query of a Postgres database (set connection in args).",
        runtime="node",
        command="npx",
        args=["-y", "@modelcontextprotocol/server-postgres"],
        needs_args=["connection_url"],
        setup_url="https://github.com/modelcontextprotocol/servers/tree/main/src/postgres",
        tier="path",
    ),
    McpServerSpec(
        id="sqlite",
        name="SQLite",
        category="Data",
        description="Read-only query of a SQLite database file.",
        runtime="python",
        command="uvx",
        args=["mcp-server-sqlite"],
        needs_args=["db_path"],
        setup_url="https://github.com/modelcontextprotocol/servers/tree/main/src/sqlite",
        tier="path",
    ),

    # ---- Maps / Location -------------------------------------------------
    McpServerSpec(
        id="google_maps",
        name="Google Maps",
        category="Search",
        description="Places, directions, geocoding via the Maps API.",
        runtime="node",
        command="npx",
        args=["-y", "@modelcontextprotocol/server-google-maps"],
        env_required=["GOOGLE_MAPS_API_KEY"],
        setup_url="https://console.cloud.google.com/google/maps-apis",
        tier="api_key",
    ),

    # ---- Thinking aids ---------------------------------------------------
    McpServerSpec(
        id="sequential_thinking",
        name="Sequential Thinking",
        category="Productivity",
        description="Structured step-by-step problem-solving scratchpad.",
        runtime="node",
        command="npx",
        args=["-y", "@modelcontextprotocol/server-sequentialthinking"],
        setup_url="https://github.com/modelcontextprotocol/servers/tree/main/src/sequentialthinking",
        tier="zero",
    ),
    McpServerSpec(
        id="time",
        name="Time / Timezone",
        category="Productivity",
        description="Current time in any timezone, conversions, date math.",
        runtime="python",
        command="uvx",
        args=["mcp-server-time"],
        setup_url="https://github.com/modelcontextprotocol/servers/tree/main/src/time",
        tier="zero",
    ),
]


# ---- tier helpers ----------------------------------------------------------

TIER_LABELS = {
    "zero":    ("Zero-setup",  "Auto-enabled, no configuration needed."),
    "path":    ("Path needed", "Tell Iris which folder/database to use, once."),
    "oauth":   ("One-click",   "Sign in via browser — no token to paste."),
    "api_key": ("Advanced",    "Requires an API token from the provider."),
}


def zero_setup_ids() -> List[str]:
    """Servers Iris auto-enables on first launch (shipped app). Filtered
    by tier='zero' AND runtime that the shipped app actually has — Node
    is NOT bundled today, so node-runtime servers are excluded from
    auto-enable even if their tier is 'zero'."""
    out: List[str] = []
    for spec in DEFAULT_SERVERS:
        if spec.tier != "zero":
            continue
        if spec.runtime == "node":
            # Bundled-Node future will flip this. Today the user would
            # have to install Node manually, so don't auto-enable and
            # silently fail.
            continue
        out.append(spec.id)
    return out


def by_id(spec_id: str) -> Optional[McpServerSpec]:
    """Look up a default-catalog entry by id. Returns None for unknown ids."""
    for spec in DEFAULT_SERVERS:
        if spec.id == spec_id:
            return spec
    return None


def categories() -> List[str]:
    """De-duplicated category list, preserving catalog order."""
    seen: Dict[str, None] = {}
    for spec in DEFAULT_SERVERS:
        if spec.category not in seen:
            seen[spec.category] = None
    return list(seen.keys())
