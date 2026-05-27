"""Chrome connector — the API-first path for controlling a running Chrome.

Wraps ChromeController (window focus + keyboard-shortcut driving). When
Chrome is running, iris does back/forward/refresh/new-tab/search with one
deterministic call instead of clicking toolbar buttons.

Scope note: this deliberately overlaps the built-in `press_hotkey` /
`open_url` / `web_navigate` tools — named verbs are easier for the model
to pick than constructing the right chord. It is only exposed when Chrome
is actually running (`is_running()`), so it adds no token cost otherwise;
launching/opening URLs cold is left to the built-in `open_url`.

Controller sharing: reuses the executor's lazily-created ChromeController
via `executor._ensure_chrome()` so there's a single instance.

Author: Konstantin Markov
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from .base import Connector, connector_result


class ChromeConnector(Connector):
    id = "chrome"

    def __init__(self, executor: Optional[Any] = None,
                 controller: Optional[Any] = None) -> None:
        self._executor = executor
        self._own = controller

    def _ctrl(self):
        if self._executor is not None:
            try:
                ctrl = self._executor._ensure_chrome()
                if ctrl is not None:
                    return ctrl
            except Exception:
                pass
        if self._own is None:
            from ...debug.chrome_controller import ChromeController
            self._own = ChromeController()
        return self._own

    def available(self) -> bool:
        try:
            c = self._ctrl()
            return bool(getattr(c, "available", False) and c.is_running())
        except Exception:
            return False

    def tools(self) -> List[Dict[str, Any]]:
        def fn(name: str, desc: str, props: Dict[str, Any] | None = None,
               required: List[str] | None = None) -> Dict[str, Any]:
            return {
                "type": "function",
                "name": name,
                "description": desc,
                "parameters": {
                    "type": "object",
                    "properties": props or {},
                    "required": required or [],
                    "additionalProperties": False,
                },
            }

        return [
            fn("chrome_back", "Navigate the active Chrome tab back."),
            fn("chrome_forward", "Navigate the active Chrome tab forward."),
            fn("chrome_refresh", "Refresh the active Chrome tab."),
            fn("chrome_new_tab", "Open a new Chrome tab."),
            fn("chrome_search",
               "Search Google in Chrome for a query (opens results in Chrome).",
               {"query": {"type": "string", "description": "What to search for."}},
               ["query"]),
            fn("chrome_open_url",
               "Open a URL in the running Chrome.",
               {"url": {"type": "string", "description": "Absolute URL to open."}},
               ["url"]),
        ]

    def execute(self, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        c = self._ctrl()
        if name == "chrome_back":
            return connector_result("ok", navigated=bool(c.navigate_back()))
        if name == "chrome_forward":
            return connector_result("ok", navigated=bool(c.navigate_forward()))
        if name == "chrome_refresh":
            return connector_result("ok", refreshed=bool(c.refresh_page()))
        if name == "chrome_new_tab":
            return connector_result("ok", opened=bool(c.new_tab()))
        if name == "chrome_search":
            query = str(args.get("query") or "").strip()
            if not query:
                return connector_result("error", error="query is required")
            return connector_result("ok" if c.search_google(query) else "error", query=query)
        if name == "chrome_open_url":
            url = str(args.get("url") or "").strip()
            if not url:
                return connector_result("error", error="url is required")
            return connector_result("ok" if c.open_url(url) else "error", url=url)
        return connector_result("error", error=f"unknown chrome tool: {name}", code="no_handler")
