"""Throwaway end-to-end check for the iris connector model.

Proves the OpenClaw-style routing without needing a live OpenAI session:
  * builds a ConnectorRegistry with the Spotify connector,
  * wraps it in ToolRegistry alongside a fake GUI/built-in executor,
  * shows the assembled tool list (built-in + spotify tools),
  * shows API-first routing: a spotify_* call hits the connector, while a
    non-connector tool (e.g. click_screen) falls through to the executor.

If Spotify is set up + authorized it also makes a real API call
(spotify_now_playing) so you can see the API path actually working.

Run:  .venv\\Scripts\\python.exe iris_connector_demo.py
Dev tool — not shipped, safe to delete.
"""
from __future__ import annotations

from src.hgr.live_api.connectors import ConnectorRegistry
from src.hgr.live_api.connectors.spotify_connector import SpotifyConnector
from src.hgr.live_api.tool_registry import ToolRegistry


class _FakeExecutor:
    """Stands in for ToolExecutor (the GUI/built-in path) so we can see
    fallback routing without spinning up the whole iris stack."""

    def execute(self, name, args):
        return {"status": "ok", "routed_to": "GUI/built-in executor", "tool": name, "args": args}


def main() -> int:
    connectors = ConnectorRegistry()
    connectors.register(SpotifyConnector())  # pass the app's controller in production
    reg = ToolRegistry(_FakeExecutor(), connectors=connectors)

    spotify = SpotifyConnector()
    print(f"Spotify connector available (set up + authorized)? {spotify.available()}")

    tools = reg.openai_tools()
    spotify_tools = [t["name"] for t in tools if t["name"].startswith("spotify_")]
    print(f"Total tools exposed to the model: {len(tools)}")
    print(f"Spotify (API) tools exposed: {spotify_tools or '(none — Spotify not authorized, GUI fallback only)'}")

    print("\nRouting check:")
    # A connector-owned tool -> goes to the Spotify connector (API path).
    r1 = reg.call("spotify_now_playing", {})
    print(f"  spotify_now_playing -> {r1}")
    # A non-connector tool -> falls through to the GUI/built-in executor.
    r2 = reg.call("click_screen", {"x": 0.5, "y": 0.5, "coordinate_space": "normalized"})
    print(f"  click_screen        -> {r2}")

    print("\nInterpretation:")
    print("  * If spotify_now_playing returned a connector result, the API-first path works.")
    print("  * click_screen routed to the GUI/built-in executor = fallback works.")
    print("  * If Spotify wasn't authorized, its tools weren't exposed and iris would")
    print("    have used GUI computer-use for Spotify instead — exactly the fallback design.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
