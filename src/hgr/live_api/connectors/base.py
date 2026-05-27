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

    def available_tool_schemas(self) -> List[Dict[str, Any]]:
        """Schemas for all currently-available connectors. Also (re)builds
        the name->connector ownership map used for routing."""
        schemas: List[Dict[str, Any]] = []
        owner: Dict[str, Connector] = {}
        for c in self._connectors:
            try:
                if not c.available():
                    continue
                for schema in c.tools():
                    name = schema.get("name")
                    if not name or name in owner:
                        continue
                    owner[name] = c
                    schemas.append(dict(schema))
            except Exception:
                # A broken connector must never take down tool assembly.
                continue
        self._owner = owner
        return schemas

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
                "error", error=f"{type(exc).__name__}: {exc}", code="connector_exception"
            )
