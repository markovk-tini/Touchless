"""Earcon dispatcher — turns ToolInvocation events into earcons.

Phase-3 wiring. Subscribes to the global InvocationBus and plays a
short non-verbal tone for each meaningful tool-call transition:

  * `STARTING` (status == "starting") — quick 'tick' so the user
    knows Iris is acting.
  * `DONE`     (status == "ok")        — soft 'chime' when something
    user-visible completes.
  * `ERROR`    (status == "error")     — gentle 'dip' on failure.
  * `NEEDS_CONFIRM` — short 'double-tap' when the safety gate is
    about to surface a confirm prompt (so the user knows to look).
  * `DECLINED` — 'down-step' when the user cancels.

The actual audio is synthesized + played by EarconPlayer (which
respects QuietMode + earcon-allowed flags). Earcons that fire
during quiet mode are silently suppressed unless the kind is in
the force-through-quiet set (NEEDS_CONFIRM, ERROR).

Tools considered "trivial" (read-only, fast, fires every few
seconds in voice mode) don't earcon their DONE — they'd be
annoying. The default skip-list matches the never-cache list +
volume_get / weather_get / etc.

Author: Konstantin Markov
"""
from __future__ import annotations

import threading
from typing import Any, Optional


# Tools whose successful completion should NOT play a DONE earcon.
# These fire frequently in background polls; the user doesn't need
# a chime every time. ERROR for these still earcons (errors are
# rare and informative).
_SKIP_DONE_TOOLS = frozenset({
    "weather_get", "volume_get", "calendar_list_events",
    "gmail_list", "ms_mail_list", "email_summary",
    "spotify_now_playing", "iris_lookup_contact",
    "phone_link_read_recent",
    # Pseudo-tools that are themselves UI affordances.
    "iris_open_last", "iris_set_preference",
    "iris_remember_contact", "iris_forget_contact",
})

# Tools that should ALWAYS earcon on done, regardless of skip-list
# overlap (catches sends, deletes, creates — the user wants the
# confirmation chime so they don't have to look at the screen).
_ALWAYS_DONE_TOOLS = frozenset({
    "gmail_send", "ms_mail_send", "outlook_send",
    "outlook_compose", "slack_post", "discord_post",
    "phone_link_send_text", "teams_send", "todo_add",
    "gdocs_create", "sheets_create", "slides_create",
    "drive_upload", "onedrive_upload", "notion_create_page",
    "notion_add_to_database", "notion_append_to_page",
    "onenote_create", "onenote_append_text",
    "file_delete",
})


class EarconDispatcher:
    """Subscribes to the InvocationBus and plays earcons per
    invocation status transition. Idempotent attach."""

    def __init__(self, *, player: Optional[Any] = None,
                 quiet_mode: Optional[Any] = None) -> None:
        self._player = player
        self._quiet = quiet_mode
        self._unsubscribe: Optional[Any] = None
        # Per-invocation_id last-status, so we earcon only on the
        # transition (the executor publishes the same ToolInvocation
        # twice: first with status="starting", then again with the
        # terminal status).
        self._lock = threading.Lock()
        self._last_status: dict = {}

    def _get_player(self):
        if self._player is None:
            from .earcons import global_earcons
            self._player = global_earcons()
        return self._player

    def attach_to_bus(self, bus: Optional[Any] = None) -> None:
        """Subscribe to the bus. Idempotent — re-attach detaches first
        so we don't double-deliver invocations."""
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

    # ---- subscriber ---------------------------------------------------

    def _on_invocation(self, inv) -> None:
        # Honor incognito: no earcons during private mode (the user
        # asked for quiet operation; tones are still a side-channel
        # leak about activity).
        try:
            extra = getattr(inv, "extra", None) or {}
            if isinstance(extra, dict) and extra.get("incognito"):
                return
        except Exception:
            return
        tool = getattr(inv, "tool", "") or ""
        status = str(getattr(inv, "status", "") or "")
        inv_id = getattr(inv, "invocation_id", None)
        if not inv_id:
            inv_id = id(inv)
        # Skip if we already played this transition.
        with self._lock:
            prev = self._last_status.get(inv_id)
            if prev == status:
                return
            self._last_status[inv_id] = status
            # Bound the dict — drop oldest if it grows past 200.
            if len(self._last_status) > 200:
                for k in list(self._last_status.keys())[:50]:
                    self._last_status.pop(k, None)
        kind = self._resolve_earcon_kind(tool, status)
        if kind is None:
            return
        try:
            self._get_player().play(kind)
        except Exception:
            pass

    def _resolve_earcon_kind(self, tool: str, status: str):
        """Map a (tool, status) pair to an EarconKind or None."""
        from .earcons import EarconKind
        if status == "starting":
            # No earcon for starting — would chime way too often.
            return None
        if status == "needs_confirmation":
            return EarconKind.NEEDS_CONFIRM
        if status == "cancelled":
            return EarconKind.DECLINED
        if status == "error":
            return EarconKind.ERROR
        if status == "ok":
            # Suppress DONE chimes while Iris is actively speaking
            # (the chime would overlap her voice audibly). The
            # NEEDS_CONFIRM and ERROR earcons are in the force-
            # through-quiet set so they bypass this check.
            if self._iris_currently_speaking():
                return None
            # User-visible side effects → DONE chime. Background
            # reads → silence.
            if tool in _ALWAYS_DONE_TOOLS:
                return EarconKind.DONE
            if tool in _SKIP_DONE_TOOLS:
                return None
            # Default: chime on done for everything else (custom
            # tools, MCP, etc.). Better noisy than silent.
            return EarconKind.DONE
        return None

    def _iris_currently_speaking(self) -> bool:
        """Best-effort: ask the manager whether a realtime response
        is currently being streamed. When unknown, default to False
        so we don't over-suppress."""
        try:
            from . import live_api_manager  # type: ignore
            # The manager is a singleton-ish process-wide object;
            # try to introspect the most recently constructed one.
            mgr = getattr(live_api_manager, "_last_constructed", None)
            if mgr is None:
                return False
            return bool(getattr(mgr, "_response_active", False))
        except Exception:
            return False


# ---- module singleton --------------------------------------------------

_dispatcher: Optional[EarconDispatcher] = None
_lock = threading.Lock()


def global_earcon_dispatcher() -> EarconDispatcher:
    global _dispatcher
    if _dispatcher is None:
        with _lock:
            if _dispatcher is None:
                _dispatcher = EarconDispatcher()
    return _dispatcher


def _reset_for_tests() -> None:
    global _dispatcher
    with _lock:
        target = _dispatcher
        _dispatcher = None
    if target is not None:
        try:
            target.detach_from_bus()
        except Exception:
            pass
