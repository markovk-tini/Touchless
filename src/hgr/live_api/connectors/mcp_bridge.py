"""MCP bridge — turn any MCP server into iris connectors.

This is how iris gets **breadth without hand-writing every connector**: an
MCP (Model Context Protocol) server exposes a set of tools, and this bridge
wraps each configured server as a `Connector` so its tools flow into the
same registry — discoverable by the capability-search router and routed
like any other connector. One bridge ≈ many apps, maintained by the MCP
ecosystem instead of by us.

Design
------
* **Dormant by default.** Needs the ``mcp`` SDK (``pip install mcp``) AND a
  server config file. Without either, ``available()`` is False, no tools are
  exposed, nothing breaks — same pattern as the Google connectors.
* **Local / self-hosted.** Servers are launched as child processes over
  stdio, so data stays on the machine (no cloud broker). Privacy-preserving.
* **Sync/async bridge.** The MCP SDK is asyncio; connector ``execute()`` is
  sync (called from the realtime worker thread). Each server gets a private
  daemon thread running its own event loop and a persistent session; sync
  calls are marshalled onto that loop and waited on.

Config: JSON at ``~/Documents/Touchless/mcp_servers.json`` (override with
``TOUCHLESS_MCP_CONFIG``)::

    {
      "servers": [
        {"name": "filesystem",
         "command": "npx",
         "args": ["-y", "@modelcontextprotocol/server-filesystem", "C:\\\\Users\\\\me"]},
        {"name": "github",
         "command": "npx", "args": ["-y", "@modelcontextprotocol/server-github"],
         "env": {"GITHUB_TOKEN": "ghp_..."}}
      ]
    }

Author: Konstantin Markov
"""
from __future__ import annotations

import json
import os
import re
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

from .base import Connector, connector_result


def libs_available() -> bool:
    try:
        import mcp  # noqa: F401
        return True
    except Exception:
        return False


def _config_path() -> Path:
    override = os.environ.get("TOUCHLESS_MCP_CONFIG")
    if override:
        return Path(override).expanduser()
    return Path.home() / "Documents" / "Touchless" / "mcp_servers.json"


def load_server_configs() -> List[Dict[str, Any]]:
    """Parse the server list from the config file. Returns [] if absent or
    malformed (bridge stays dormant)."""
    path = _config_path()
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    servers = data.get("servers") if isinstance(data, dict) else None
    if not isinstance(servers, list):
        return []
    out = []
    for s in servers:
        if isinstance(s, dict) and s.get("name") and s.get("command"):
            out.append(s)
    return out


def _sanitize(text: str) -> str:
    """Make a string safe for an OpenAI tool-name segment ([a-z0-9_])."""
    return re.sub(r"[^a-z0-9_]+", "_", (text or "").lower()).strip("_") or "x"


class _ServerSession:
    """Owns one MCP server connection on a private asyncio loop thread.

    The MCP SDK's stdio_client / ClientSession are async context managers
    that must stay open for the connection's life, so we enter them once on
    the loop and keep them; sync callers submit coroutines via run().
    """

    def __init__(self, cfg: Dict[str, Any]) -> None:
        self._cfg = cfg
        self._loop = None
        self._thread: Optional[threading.Thread] = None
        self._session = None
        self._tools: List[Dict[str, Any]] = []  # OpenAI schemas (un-namespaced names)
        self._started = False
        self._ok = False
        self._lock = threading.Lock()
        self._stack = None  # AsyncExitStack holding the open contexts

    # ---- public (sync) -------------------------------------------------
    def ensure_started(self) -> bool:
        with self._lock:
            if self._started:
                return self._ok
            self._started = True
            if not libs_available():
                return False
            import asyncio

            self._loop = asyncio.new_event_loop()
            self._thread = threading.Thread(
                target=self._run_loop, name=f"MCP-{self._cfg.get('name')}", daemon=True)
            self._thread.start()
            try:
                fut = asyncio.run_coroutine_threadsafe(self._connect(), self._loop)
                self._ok = bool(fut.result(timeout=30))
            except Exception:
                self._ok = False
            return self._ok

    def tools(self) -> List[Dict[str, Any]]:
        return list(self._tools) if self._ok else []

    def call(self, tool_name: str, args: Dict[str, Any], timeout: float = 60.0) -> Dict[str, Any]:
        if not self._ok or self._loop is None:
            return connector_result("error", error="mcp server not connected", code="not_ready")
        import asyncio
        try:
            fut = asyncio.run_coroutine_threadsafe(
                self._call_tool(tool_name, args or {}), self._loop)
            return fut.result(timeout=timeout)
        except Exception as exc:
            return connector_result("error", error=f"{type(exc).__name__}: {exc}")

    # ---- loop-thread internals ----------------------------------------
    def _run_loop(self) -> None:
        self._loop.run_forever()

    async def _connect(self) -> bool:
        from contextlib import AsyncExitStack
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        params = StdioServerParameters(
            command=str(self._cfg.get("command")),
            args=list(self._cfg.get("args") or []),
            env={**os.environ, **(self._cfg.get("env") or {})},
        )
        self._stack = AsyncExitStack()
        read, write = await self._stack.enter_async_context(stdio_client(params))
        self._session = await self._stack.enter_async_context(ClientSession(read, write))
        await self._session.initialize()
        listing = await self._session.list_tools()
        self._tools = [self._to_openai_schema(t) for t in (listing.tools or [])]
        return True

    async def _call_tool(self, tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        result = await self._session.call_tool(tool_name, args)
        # Flatten text content blocks into a single string for the model.
        chunks = []
        for block in (getattr(result, "content", None) or []):
            text = getattr(block, "text", None)
            if text:
                chunks.append(text)
        is_error = bool(getattr(result, "isError", False))
        return connector_result(
            "error" if is_error else "ok",
            output="\n".join(chunks) if chunks else None,
        )

    @staticmethod
    def _to_openai_schema(tool: Any) -> Dict[str, Any]:
        schema = getattr(tool, "inputSchema", None) or {
            "type": "object", "properties": {}, "additionalProperties": False}
        return {
            "type": "function",
            "name": getattr(tool, "name", "tool"),
            "description": getattr(tool, "description", "") or "",
            "parameters": schema,
        }


class MCPConnector(Connector):
    """Wraps one MCP server. Tools are namespaced ``mcp_<server>_<tool>`` to
    avoid colliding with built-ins or other connectors."""

    def __init__(self, cfg: Dict[str, Any]) -> None:
        self._cfg = cfg
        self.id = "mcp_" + _sanitize(cfg.get("name", "server"))
        self._prefix = self.id + "_"
        self._session = _ServerSession(cfg)
        self.description = (
            f"{cfg.get('name')} MCP server: "
            + (cfg.get("description") or "external tools via Model Context Protocol")
        )

    def available(self) -> bool:
        try:
            return bool(self._session.ensure_started()) and bool(self._session.tools())
        except Exception:
            return False

    def tools(self) -> List[Dict[str, Any]]:
        out = []
        for schema in self._session.tools():
            s = dict(schema)
            s["name"] = self._prefix + _sanitize(schema.get("name", "tool"))
            out.append(s)
        return out

    def execute(self, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        if not name.startswith(self._prefix):
            return connector_result("error", error=f"not an {self.id} tool: {name}",
                                    code="no_handler")
        # Map the namespaced name back to the server's real tool name.
        suffix = name[len(self._prefix):]
        for schema in self._session.tools():
            if _sanitize(schema.get("name", "")) == suffix:
                return self._session.call(schema.get("name"), args)
        return connector_result("error", error=f"unknown mcp tool: {name}", code="no_handler")


def build_mcp_connectors() -> List[MCPConnector]:
    """One MCPConnector per configured server. Empty if the sdk/config are
    absent, so the registry simply gets nothing extra."""
    if not libs_available():
        return []
    out: List[MCPConnector] = []
    for cfg in load_server_configs():
        try:
            out.append(MCPConnector(cfg))
        except Exception:
            continue
    return out
