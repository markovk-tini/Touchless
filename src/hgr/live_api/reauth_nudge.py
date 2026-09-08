"""Reauth-nudge bus subscriber.

Phase-3 polish. When any tool returns an `auth_revoked` error, this
subscriber maps the failed tool to its owning connector and surfaces
a one-click "Reconnect X" action chip via the manager's
`suggested_actions` signal — instead of the user having to wait for
the proactive_nudge cooldown (5 errors in 30 min) or read the raw
error.

Per-connector cooldown so a single auth_revoked doesn't fire the
chip three times in one turn (the planner reviser may retry the
same call). 5 min cooldown per (connector_id) is plenty for the
real "session" use case.

Design: pure subscriber over the InvocationBus, completely decoupled
from the manager — just emits a notify-callback the manager wires.

Author: Konstantin Markov
"""
from __future__ import annotations

import threading
import time
from typing import Any, Callable, Dict, Optional


# Map a tool name (prefix or exact) to the UI action id the chip
# system already knows about. Add new mappings as connectors land.
_TOOL_TO_ACTION: Dict[str, str] = {
    "gmail_send": "connect_gmail",
    "gmail_list": "connect_gmail",
    "gmail_read": "connect_gmail",
    "gmail_compose": "connect_gmail",
    "ms_mail_send": "connect_ms",
    "ms_mail_list": "connect_ms",
    "ms_mail_read": "connect_ms",
    "ms_mail_search": "connect_ms",
    "outlook_compose": "connect_ms",
    "outlook_send": "connect_ms",
    "teams_send": "connect_ms",
    "teams_post": "connect_ms",
    "calendar_list_events": "connect_ms",
    "calendar_create_event": "connect_ms",
    "ms_calendar_create": "connect_ms",
    "drive_upload": "connect_gmail",   # google auth
    "drive_list": "connect_gmail",
    "gdocs_create": "connect_gmail",
    "sheets_create": "connect_gmail",
    "slides_create": "connect_gmail",
    "tasks_list": "connect_gmail",
    "tasks_add": "connect_gmail",
    "tasks_complete": "connect_gmail",
    "tasks_delete": "connect_gmail",
    "youtube_my_playlists": "connect_gmail",
    "youtube_playlist_items": "connect_gmail",
    "youtube_subscriptions": "connect_gmail",
    "photos_upload": "connect_gmail",
}


_PER_CONNECTOR_COOLDOWN_SEC = 300.0


ActionNotifier = Callable[[list], None]
"""Callback signature: notifier(['connect_gmail', ...]). The chip
rendering layer turns the action ids into clickable chips."""


class ReauthNudger:
    """Bus subscriber: catches `auth_revoked` errors and emits a
    reconnect-action chip via the supplied notifier."""

    def __init__(self, *, notifier: ActionNotifier) -> None:
        self._notifier = notifier
        self._lock = threading.RLock()
        self._last_fire: Dict[str, float] = {}
        self._unsubscribe: Optional[Any] = None

    def attach_to_bus(self, bus: Optional[Any] = None) -> None:
        self.detach_from_bus()
        from .tool_invocation import global_bus
        target = bus if bus is not None else global_bus()
        self._unsubscribe = target.subscribe(self._on_invocation)

    def detach_from_bus(self) -> None:
        if self._unsubscribe is not None:
            try:
                self._unsubscribe()
            except Exception:
                pass
            self._unsubscribe = None

    def _on_invocation(self, inv) -> None:
        # Honor incognito.
        try:
            extra = getattr(inv, "extra", None) or {}
            if isinstance(extra, dict) and extra.get("incognito"):
                return
        except Exception:
            return
        if str(getattr(inv, "status", "")) != "error":
            return
        err = str(getattr(inv, "error", "") or "")
        # Classify via reliability_ledger's classifier so we share
        # one source of truth across modules.
        try:
            from .reliability_ledger import classify_error
            ec = classify_error(err)
        except Exception:
            ec = "other"
        if ec not in ("auth_revoked", "not_connected", "scope_missing"):
            return
        tool = str(getattr(inv, "tool", "") or "")
        action_id = self._resolve_action(tool)
        if not action_id:
            return
        now = time.time()
        with self._lock:
            last = self._last_fire.get(action_id, 0.0)
            if (now - last) < _PER_CONNECTOR_COOLDOWN_SEC:
                return
            self._last_fire[action_id] = now
        try:
            self._notifier([action_id])
        except Exception:
            pass

    @staticmethod
    def _resolve_action(tool: str) -> str:
        if not tool:
            return ""
        # Exact match first.
        if tool in _TOOL_TO_ACTION:
            return _TOOL_TO_ACTION[tool]
        # Prefix match (notion_, slack_, etc. → no UI action yet,
        # but space for the future).
        return ""

    def reset(self) -> None:
        with self._lock:
            self._last_fire.clear()
