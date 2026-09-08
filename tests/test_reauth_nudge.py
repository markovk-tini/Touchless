"""Tests for ReauthNudger (Phase 3 polish)."""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from hgr.live_api.reauth_nudge import (  # noqa: E402
    ReauthNudger, _TOOL_TO_ACTION,
)
from hgr.live_api.tool_invocation import (  # noqa: E402
    InvocationBus, InvocationSource, ToolInvocation,
)


def _inv(tool: str, status: str = "error",
         error: str = "HTTP 401 Unauthorized",
         incognito: bool = False,
         invocation_id: str = "inv") -> ToolInvocation:
    inv = ToolInvocation(
        invocation_id=invocation_id, turn_id="t", tool=tool,
        args={}, source=InvocationSource.PLANNER, status=status,
        error=error)
    if incognito:
        inv.extra["incognito"] = True
    return inv


def _new_nudger():
    fired = []
    n = ReauthNudger(notifier=lambda actions: fired.append(actions))
    return n, fired


# ---- happy paths -------------------------------------------------------

def test_gmail_send_auth_revoked_fires_connect_gmail():
    n, fired = _new_nudger()
    n._on_invocation(_inv("gmail_send", error="401 Unauthorized"))
    assert fired == [["connect_gmail"]]


def test_outlook_compose_auth_revoked_fires_connect_ms():
    n, fired = _new_nudger()
    n._on_invocation(_inv("outlook_compose", error="invalid_grant"))
    assert fired == [["connect_ms"]]


def test_teams_send_maps_to_ms():
    n, fired = _new_nudger()
    n._on_invocation(_inv("teams_send", error="401 unauthorized"))
    assert fired == [["connect_ms"]]


def test_gdocs_create_maps_to_gmail_auth():
    # Drive/Docs share Google OAuth with Gmail in our app.
    n, fired = _new_nudger()
    n._on_invocation(_inv("gdocs_create", error="auth_revoked: token expired"))
    assert fired == [["connect_gmail"]]


def test_not_connected_error_also_fires():
    n, fired = _new_nudger()
    n._on_invocation(_inv("gmail_send",
                          error="precondition_not_met: not connected"))
    assert fired == [["connect_gmail"]]


# ---- filtering ---------------------------------------------------------

def test_ok_status_does_not_fire():
    n, fired = _new_nudger()
    n._on_invocation(_inv("gmail_send", status="ok",
                          error=""))
    assert fired == []


def test_other_error_class_does_not_fire():
    n, fired = _new_nudger()
    n._on_invocation(_inv("gmail_send",
                          error="connection timed out"))
    # transient_network classification → no reauth chip.
    assert fired == []


def test_unknown_tool_does_not_fire():
    n, fired = _new_nudger()
    n._on_invocation(_inv("some_unknown_tool",
                          error="HTTP 401"))
    assert fired == []


def test_incognito_suppresses():
    n, fired = _new_nudger()
    n._on_invocation(_inv("gmail_send", error="HTTP 401",
                          incognito=True))
    assert fired == []


# ---- cooldown ----------------------------------------------------------

def test_per_connector_cooldown_blocks_repeat():
    n, fired = _new_nudger()
    n._on_invocation(_inv("gmail_send", error="HTTP 401",
                          invocation_id="i1"))
    n._on_invocation(_inv("gmail_send", error="HTTP 401",
                          invocation_id="i2"))
    n._on_invocation(_inv("gmail_list", error="HTTP 401",
                          invocation_id="i3"))
    # All three map to connect_gmail; should only fire once.
    assert fired == [["connect_gmail"]]


def test_distinct_connectors_each_fire():
    n, fired = _new_nudger()
    n._on_invocation(_inv("gmail_send", error="HTTP 401",
                          invocation_id="i1"))
    n._on_invocation(_inv("ms_mail_send", error="HTTP 401",
                          invocation_id="i2"))
    assert fired == [["connect_gmail"], ["connect_ms"]]


# ---- attach / detach ---------------------------------------------------

def test_attach_to_bus_is_idempotent():
    n, fired = _new_nudger()
    bus = InvocationBus()
    n.attach_to_bus(bus)
    n.attach_to_bus(bus)
    bus.publish(_inv("gmail_send", error="HTTP 401"))
    assert fired == [["connect_gmail"]]


def test_detach_stops_delivery():
    n, fired = _new_nudger()
    bus = InvocationBus()
    n.attach_to_bus(bus)
    n.detach_from_bus()
    bus.publish(_inv("gmail_send", error="HTTP 401"))
    assert fired == []


# ---- helpers -----------------------------------------------------------

def test_action_map_covers_main_providers():
    # Sanity: ensure the most-common send tools have mappings.
    must_map = {"gmail_send", "ms_mail_send", "outlook_compose",
                "teams_send", "drive_upload"}
    for tool in must_map:
        assert tool in _TOOL_TO_ACTION


def test_action_map_covers_round2_google_tools():
    # Round-2 Google connectors (tasks/youtube/photos) must route to
    # connect_gmail so scope_missing on them shows the reconnect chip.
    must_map = {"tasks_add", "tasks_list", "tasks_complete", "tasks_delete",
                "youtube_my_playlists", "youtube_playlist_items",
                "youtube_subscriptions", "photos_upload"}
    for tool in must_map:
        assert _TOOL_TO_ACTION.get(tool) == "connect_gmail"


def test_scope_missing_error_fires_chip():
    # The scope_missing connector_result error text + code should fire
    # the reconnect chip the same way auth_revoked does.
    n, fired = _new_nudger()
    n._on_invocation(_inv(
        "tasks_add",
        error=("I need Google Tasks access (Google scope "
               "'https://www.googleapis.com/auth/tasks') — please click "
               "'Connect Google' in Settings to re-authorize. The "
               "consent screen will include it.")))
    assert fired == [["connect_gmail"]]


def test_reset_clears_cooldowns():
    n, fired = _new_nudger()
    n._on_invocation(_inv("gmail_send", error="HTTP 401"))
    n.reset()
    n._on_invocation(_inv("gmail_send", error="HTTP 401",
                          invocation_id="i2"))
    assert fired == [["connect_gmail"], ["connect_gmail"]]
