"""Connector base class + registry.

The router contract is deliberately tiny so connectors stay cheap to
write and the per-call overhead stays in the microseconds:

    Connector.tools()      -> [OpenAI tool schema, ...]   (what the model sees)
    Connector.available()  -> bool                         (expose its tools?)
    Connector.execute(name, args) -> result dict           (run a tool)

ConnectorRegistry merges the *available* connectors' schemas into one
list for `session.update.tools`, and routes a tool call to its owning
connector. A tool that no connector owns returns None from execute(),
which the caller (ToolRegistry) treats as "fall through to the GUI
computer-use executor."
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional


def connector_result(status: str = "ok", **fields: Any) -> Dict[str, Any]:
    """Match the executor's result shape so model output is uniform."""
    out: Dict[str, Any] = {"status": status}
    out.update(fields)
    return out


def friendly_api_error(exc: Exception, *, api_label: str = "The service") -> str:
    """Translate an HTTP client exception into a spoken-friendly message.

    googleapiclient.HttpError.__str__ (and most requests-style errors)
    dumps the entire response body including a JSON blob. Passing that
    to the user unfiltered produces a wall of text that reads badly in
    chat and is impossible aloud. This helper extracts the status code
    and (for Google APIs) the machine-readable 'reason' string
    (SERVICE_DISABLED, PERMISSION_DENIED, ...) and returns one
    conversational sentence with an actionable next step. Falls back to
    type+summary for non-HTTP exceptions.

    api_label: human-readable name of the failing API ("Google
    Contacts", "Gmail", "Phone Link") — surfaces in the message so the
    user knows which integration hit the wall.
    """
    status = None
    reason = None
    try:
        resp = getattr(exc, "resp", None)
        if resp is not None:
            status = int(getattr(resp, "status", 0)) or None
    except Exception:
        status = None
    # Parse Google's error detail JSON for the machine-readable reason.
    try:
        import json as _json
        content = getattr(exc, "content", None)
        if content:
            body = _json.loads(
                content.decode("utf-8") if isinstance(content, bytes)
                else content)
            errors = ((body or {}).get("error") or {}).get("errors") or []
            if errors:
                reason = (errors[0] or {}).get("reason")
            details = ((body or {}).get("error") or {}).get("details") or []
            for d in details:
                if isinstance(d, dict) and d.get("reason"):
                    reason = d.get("reason")
                    break
    except Exception:
        reason = None
    if status == 403 and reason == "SERVICE_DISABLED":
        return (f"{api_label} isn't enabled for this project yet. "
                f"Turn it on in Google Cloud Console and try again in "
                f"a minute.")
    if status == 403:
        return (f"{api_label} refused the request — probably a missing "
                f"scope or permission. Try re-authorizing.")
    if status == 404:
        return f"{api_label} couldn't find that item."
    if status == 401:
        return f"{api_label} says the session expired — re-authorize."
    if status == 429:
        return f"{api_label} is rate-limiting me. Give it a minute."
    if status and status >= 500:
        return f"{api_label} had a server-side hiccup. Worth a retry."
    return f"{api_label} error: {type(exc).__name__}"


class Connector:
    """Base class for an iris capability connector."""

    id: str = "connector"
    # One-line, synonym-rich summary used by the capability-search router
    # to match a connector to a natural-language task. Set per connector
    # (or assigned centrally in build_connector_registry).
    description: str = ""

    def tools(self) -> List[Dict[str, Any]]:
        """OpenAI tool schemas this connector exposes. Override."""
        raise NotImplementedError

    def available(self) -> bool:
        """Whether this connector is usable right now (configured/authed).
        When False, its tools are not exposed to the model — so the model
        won't try the API path and will use the GUI fallback instead."""
        return True

    def execute(self, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        """Run a tool this connector owns; return a result dict. Override."""
        raise NotImplementedError


class ConnectorRegistry:
    def __init__(self) -> None:
        self._connectors: List[Connector] = []
        # tool name -> owning connector, rebuilt whenever the tool list is
        # assembled (availability can change between sessions, e.g. auth).
        self._owner: Dict[str, Connector] = {}

    def register(self, connector: Connector) -> None:
        self._connectors.append(connector)

    def find_by_id(self, name: str) -> Optional[Connector]:
        """Look up a connector by its `id` attribute (case-insensitive).
        Used by the iris_setup_tool pseudo-tool to invoke setup_self() on
        the right connector after a 'set up X' command."""
        norm = (name or "").strip().lower()
        if not norm:
            return None
        # Try exact match first, then substring (so 'kicad' finds
        # 'kicad_cli', 'ms' finds 'ms365').
        for c in self._connectors:
            if (getattr(c, "id", "") or "").lower() == norm:
                return c
        for c in self._connectors:
            cid = (getattr(c, "id", "") or "").lower()
            if norm in cid or cid in norm:
                return c
        return None

    def is_available_for(self, name: str) -> Optional[str]:
        """None when the connector owning `name` is runnable right now,
        else a short human-readable reason ('outlook not connected', etc.).
        Forces an ownership-map rebuild on miss so callers don't have to
        pre-query schemas to get a meaningful answer."""
        owner = self._owner.get(name)
        if owner is None:
            self.available_tool_schemas()
            owner = self._owner.get(name)
        if owner is None:
            return None  # no connector owns it — built-in path, not our call
        try:
            if not owner.available():
                cid = getattr(owner, "id", "connector")
                return f"{cid} not connected"
        except Exception as exc:
            return f"availability check failed: {type(exc).__name__}"
        return None

    def available_tool_schemas(
        self, *, exclude_lazy: bool = False,
    ) -> List[Dict[str, Any]]:
        """Schemas for all currently-available connectors. Also (re)builds
        the name->connector ownership map used for routing.

        `exclude_lazy=True` filters out connectors flagged for on-demand
        loading (currently any MCP connector, since `mcp_*` catalogs can
        balloon to hundreds of tools — those load via find_capability
        instead). Ownership map is still built across ALL connectors so
        a lazy-loaded tool can be routed once it has been exposed."""
        schemas: List[Dict[str, Any]] = []
        owner: Dict[str, Connector] = {}
        for c in self._connectors:
            try:
                if not c.available():
                    continue
                is_lazy = self._is_lazy(c)
                for schema in c.tools():
                    name = schema.get("name")
                    if not name or name in owner:
                        continue
                    owner[name] = c
                    if exclude_lazy and is_lazy:
                        continue
                    schemas.append(dict(schema))
            except Exception:
                # A broken connector must never take down tool assembly.
                continue
        self._owner = owner
        return schemas

    @staticmethod
    def _is_lazy(connector: "Connector") -> bool:
        """True if this connector's tools should be lazy-loaded (via the
        find_capability meta-tool) instead of eagerly exposed in the
        initial session.update. MCP bridges qualify because each server
        can register dozens of tools; eagerly exposing them all would
        bloat the realtime model's context and degrade tool-call
        accuracy. Detected by `id` prefix to keep the contract narrow."""
        cid = (getattr(connector, "id", "") or "").lower()
        return cid.startswith("mcp_")

    def catalog(self) -> List[Dict[str, Any]]:
        """For each *available* connector, its id/description and the tool
        names it would expose. Lets the router show the model what API
        fast-paths exist without putting every full schema in-context."""
        out: List[Dict[str, Any]] = []
        for c in self._connectors:
            try:
                if not c.available():
                    continue
                tools = c.tools()
            except Exception:
                continue
            out.append({
                "id": getattr(c, "id", "connector"),
                "description": getattr(c, "description", "") or "",
                "tools": [t.get("name") for t in tools if t.get("name")],
            })
        return out

    def search(self, query: str, limit: int = 3) -> List[Dict[str, Any]]:
        """Rank *available* connectors by relevance to a task description.
        Returns [{id, description, score, schemas:[full tool schemas]}] so
        the caller can load the winning connector's tools into the session.
        Also (re)builds the ownership map so the loaded tools route."""
        import re

        # Drop non-discriminative words so a task like "defragment the flux
        # capacitor" doesn't match a connector just because it says "the".
        stop = {
            "the", "a", "an", "to", "of", "on", "in", "at", "my", "me", "your",
            "you", "it", "is", "are", "and", "or", "for", "with", "this", "that",
            "some", "any", "please", "can", "could", "would", "should", "i",
            "we", "us", "be", "do", "does", "did", "from", "into", "up", "down",
            "out", "as", "by", "if", "so", "then", "now", "just", "let", "lets",
        }
        terms = [t for t in re.findall(r"[a-z0-9]+", (query or "").lower())
                 if len(t) > 1 and t not in stop]
        # Rebuild ownership so anything we surface here is routable.
        self.available_tool_schemas()
        scored: List[tuple] = []
        for c in self._connectors:
            try:
                if not c.available():
                    continue
                schemas = c.tools()
            except Exception:
                continue
            cid = getattr(c, "id", "") or ""
            hay = " ".join([
                cid,
                getattr(c, "description", "") or "",
                " ".join(f"{s.get('name', '')} {s.get('description', '')}" for s in schemas),
            ]).lower()
            # Word-level match so "what" doesn't hit "whatever", etc.
            hay_words = set(re.findall(r"[a-z0-9]+", hay))
            score = sum(1 for t in terms if t in hay_words)
            if cid and cid in terms:
                score += 5  # exact app-name mention is a strong signal
            if score > 0:
                scored.append((score, c, schemas))
        scored.sort(key=lambda x: x[0], reverse=True)
        out: List[Dict[str, Any]] = []
        for score, c, schemas in scored[:limit]:
            out.append({
                "id": getattr(c, "id", "connector"),
                "description": getattr(c, "description", "") or "",
                "score": score,
                "schemas": schemas,
            })
        return out

    def handles(self, name: str) -> bool:
        if name in self._owner:
            return True
        # Lazy build if the ownership map hasn't been populated yet
        # (e.g. execute() called before the tool list was assembled).
        if not self._owner and self._connectors:
            self.available_tool_schemas()
        return name in self._owner

    def execute(self, name: str, args: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Route to the owning connector. None => no connector owns it,
        so the caller should fall through to the GUI/built-in executor."""
        connector = self._owner.get(name)
        if connector is None:
            return None
        try:
            return connector.execute(name, args or {})
        except Exception as exc:  # pragma: no cover - defensive
            return connector_result(
                "error",
                error=friendly_api_error(
                    exc, api_label=f"The {connector.id} connector"),
                code="connector_exception",
            )
