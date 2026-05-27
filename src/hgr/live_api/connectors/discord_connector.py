"""Discord connector — the API-first path for Discord voice control.

Wraps DiscordController (Discord RPC over OAuth). There is *no* built-in
GUI tool for these — toggling mute/deafen or switching voice channels by
clicking Discord's UI is fragile, so this connector is pure upside: one
RPC call, no screenshots.

Only exposed when Discord is set up and authorized (`has_authorization`),
matching the SpotifyConnector pattern. The executor does not own a
DiscordController, so this lazy-creates its own — it reads the same saved
tokens, so auth state is shared via disk.

Author: Konstantin Markov
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from .base import Connector, connector_result


class DiscordConnector(Connector):
    id = "discord"

    def __init__(self, controller: Optional[Any] = None) -> None:
        self._controller = controller

    def _ctrl(self):
        if self._controller is None:
            from ...debug.discord_controller import DiscordController
            self._controller = DiscordController()
        return self._controller

    def available(self) -> bool:
        try:
            c = self._ctrl()
            # `available` and `has_authorization` are both @property.
            return bool(getattr(c, "available", False) and getattr(c, "has_authorization", False))
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
            fn("discord_voice_status",
               "Get the Discord voice state: whether you're muted/deafened."),
            fn("discord_mute",
               "Mute or unmute yourself in Discord voice.",
               {"muted": {"type": "boolean", "description": "True to mute."}},
               ["muted"]),
            fn("discord_toggle_mute", "Toggle your Discord self-mute."),
            fn("discord_deafen",
               "Deafen or undeafen yourself in Discord voice.",
               {"deafened": {"type": "boolean", "description": "True to deafen."}},
               ["deafened"]),
            fn("discord_toggle_deafen", "Toggle your Discord self-deafen."),
            fn("discord_join_voice",
               "Join a Discord voice channel by name.",
               {"channel_name": {"type": "string",
                                 "description": "Voice channel name to join."}},
               ["channel_name"]),
            fn("discord_leave_voice", "Leave the current Discord voice channel."),
        ]

    def execute(self, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        c = self._ctrl()
        if name == "discord_voice_status":
            state = c.get_voice_settings()
            if state is None:
                return connector_result("error", error="could not read Discord voice settings")
            return connector_result(
                "ok",
                muted=bool(getattr(state, "muted", False)),
                deafened=bool(getattr(state, "deafened", False)),
            )
        if name == "discord_mute":
            muted = bool(args.get("muted"))
            return connector_result("ok" if c.set_self_mute(muted) else "error", muted=muted)
        if name == "discord_toggle_mute":
            return connector_result("ok", changed=bool(c.toggle_self_mute()))
        if name == "discord_deafen":
            deafened = bool(args.get("deafened"))
            return connector_result("ok" if c.set_self_deafen(deafened) else "error", deafened=deafened)
        if name == "discord_toggle_deafen":
            return connector_result("ok", changed=bool(c.toggle_self_deafen()))
        if name == "discord_join_voice":
            channel_name = str(args.get("channel_name") or "").strip()
            if not channel_name:
                return connector_result("error", error="channel_name is required")
            channel = c.find_voice_channel_by_name(channel_name)
            if not channel:
                return connector_result("error", error=f"no voice channel named '{channel_name}'",
                                        code="not_found")
            channel_id = channel.get("id") if isinstance(channel, dict) else None
            if not channel_id:
                return connector_result("error", error="resolved channel has no id")
            ok = bool(c.select_voice_channel(channel_id))
            return connector_result("ok" if ok else "error", channel=channel_name, joined=ok)
        if name == "discord_leave_voice":
            return connector_result("ok", left=bool(c.leave_voice_channel()))
        return connector_result("error", error=f"unknown discord tool: {name}", code="no_handler")
