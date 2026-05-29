"""Mapping from tool name -> python callable for the Live API session.

This is intentionally a thin wrapper over `ToolExecutor` so the
schemas (`schemas.py`) and the implementation (`tool_executor.py`) can
evolve independently. The registry knows which tool names are
risky-by-default and helps build the system prompt that lists all
tools to the model.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from .connectors import ConnectorRegistry
from .schemas import all_tool_schemas


# Tools that always require a confirmation overlay before executing.
# (`ask_user_confirmation` itself is the dialog; it can't gate itself.)
RISKY_TOOLS = {
    "press_hotkey",  # alt+f4, ctrl+w, etc. — shown but not auto-blocked
}

# Tools that are read-only and never need confirmation.
READ_ONLY_TOOLS = {
    "get_screen_context",
    "ask_user_confirmation",
}


class ToolRegistry:
    def __init__(
        self,
        executor: "Any",
        connectors: Optional[ConnectorRegistry] = None,
    ) -> None:
        # `executor` is `tool_executor.ToolExecutor`. Typed as Any to avoid
        # a circular import — registry is imported by the executor too.
        self._executor = executor
        # API-first connectors (Spotify, etc.). When empty, behaviour is
        # identical to before: openai_tools() == the built-in schemas and
        # call() routes straight to the GUI/built-in executor.
        self._connectors = connectors or ConnectorRegistry()

    def openai_tools(self) -> List[Dict[str, Any]]:
        # Built-in (incl. GUI computer-use) tools + available connectors'
        # API tools. The model sees both and prefers the API tool when one
        # matches (helped by system-prompt guidance), falling back to GUI.
        return all_tool_schemas() + self._connectors.available_tool_schemas()

    def names(self) -> List[str]:
        return [s["name"] for s in self.openai_tools()]

    # ---- capability-search router -------------------------------------
    def search_connectors(self, query: str, limit: int = 3) -> List[Dict[str, Any]]:
        """Rank available connectors for a task. See ConnectorRegistry.search."""
        return self._connectors.search(query, limit=limit)

    def connector_catalog(self) -> List[Dict[str, Any]]:
        return self._connectors.catalog()

    def handles_connector(self, name: str) -> bool:
        """True if an API connector owns this tool (vs the built-in/GUI
        executor). Used to label which layer executed a command."""
        return self._connectors.handles(name)

    def find_connector(self, name: str) -> Optional[Any]:
        """Look up a connector by id substring. Used by iris_setup_tool
        ('set up kicad') to find the right connector and invoke
        setup_self()."""
        return self._connectors.find_by_id(name)

    def is_available(self, name: str) -> Optional[str]:
        """Precondition check: None when the tool is runnable now, else a
        short human-readable reason (e.g. 'ms_graph not connected'). Used by
        the planner Executor to short-circuit a step before calling — gives
        clean errors like 'Outlook not connected for the active account'
        instead of cryptic auth failures from inside the connector."""
        if self._connectors.handles(name):
            return self._connectors.is_available_for(name)
        # Built-in / GUI tools — we trust the executor's own runtime checks.
        return None

    def call(self, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        # API-first: a connector that owns this tool handles it directly.
        if self._connectors.handles(name):
            result = self._connectors.execute(name, args)
            if result is not None:
                return result
        # Universal fallback: the GUI computer-use / built-in executor.
        return self._executor.execute(name, args)

    def callable_for(self, name: str) -> Callable[[Dict[str, Any]], Dict[str, Any]]:
        return lambda args: self.call(name, args)

# Author: Konstantin Markov
