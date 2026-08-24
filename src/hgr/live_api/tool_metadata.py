"""Tool-destructiveness metadata + cost tiers.

The single source of truth for "what kind of action is this tool?". Every
downstream Phase-1 trust feature reads from here:

  * Universal undo speed-bump — gates DESTRUCTIVE/IRREVERSIBLE tools.
  * Confirmation gate — auto-confirms READ/WRITE, requires user OK for
    DESTRUCTIVE, blocks IRREVERSIBLE without typed confirm.
  * Activity pill colors / icons — distinguishes "Iris read the screen"
    from "Iris just sent an email".
  * Audit log filtering / redaction.
  * Cost-aware model ladder — picks cheaper model when downstream tool
    cost_tier is high.
  * Two-tier transcription — fast-path for READ, verify pass for
    DESTRUCTIVE.

The default for unregistered tools is intentionally conservative (WRITE +
reversible=False) so a future tool I forget to backfill doesn't slip
through as if it were a screen-read.

Author: Konstantin Markov
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional


class Destructiveness(str, Enum):
    """How harmful is a tool's worst-case outcome?

    READ          No observable side effects. Re-runnable indefinitely.
                  e.g. read_screen, get_active_window, list_files,
                  weather_get, ms_mail_list (reads inbox).

    WRITE         Reversible local side effects. Changes the user's PC
                  state but can be undone (recycle bin, file restore,
                  preference rollback). e.g. clipboard_write, volume_set,
                  outlook_compose (drafts only — not sent), open_app.

    DESTRUCTIVE   External or hard-to-reverse side effects with observable
                  impact OUTSIDE the user's machine (sent message,
                  purchased something, posted publicly). e.g. gmail_send,
                  ms_mail_send, slack post, drive_upload (now visible to
                  collaborators), calendar invite sent.

    IRREVERSIBLE  Cannot be auto-undone by Iris. e.g. delete_file (after
                  recycle-bin emptied), drive_delete, payment, sending an
                  irrevocable cryptographic signature, calling an
                  external API with a charge / rate-limit consumed.

    The four tiers are deliberately coarse — the ToolMeta below carries
    the orthogonal `reversible` flag for nuance (a DESTRUCTIVE send is
    reversibly auditable but not undoable; an IRREVERSIBLE delete is
    not even auditable past the recycle-bin window).
    """
    READ = "read"
    WRITE = "write"
    DESTRUCTIVE = "destructive"
    IRREVERSIBLE = "irreversible"


@dataclass(frozen=True)
class ToolMeta:
    destructiveness: Destructiveness
    # True when Iris itself can roll back the side effect (Windows
    # recycle-bin restore counts as reversible; an email that's been
    # delivered does not).
    reversible: bool = True
    # 0 = free (deterministic / local)
    # 1 = cheap LLM (Haiku / Qwen-3B local)
    # 2 = mid LLM (Sonnet / GPT-4-class)
    # 3 = expensive (realtime / multi-step plan / vision)
    cost_tier: int = 0
    # Human-readable grouping for the activity pill / audit UI.
    category: str = ""


# Default for unknown tools. Conservative — assume potentially
# observable, NOT auto-reversible. A missing entry should err on the
# side of triggering the confirm gate, not bypassing it.
DEFAULT_META = ToolMeta(
    destructiveness=Destructiveness.WRITE,
    reversible=False,
    cost_tier=0,
    category="unknown",
)


# ---- Canonical metadata for built-in + connector tools --------------------
#
# Source of truth: as the project ships new tools, add an entry here.
# Backfilled across ~80 known tools below. The MCP bridge inherits
# `DEFAULT_META` until per-server inspection in Phase 2.

DEFAULT_METADATA: Dict[str, ToolMeta] = {
    # ---------- screen / perception (all READ) ----------
    "get_screen_context": ToolMeta(Destructiveness.READ, True, 1, "screen"),
    "read_screen": ToolMeta(Destructiveness.READ, True, 1, "screen"),
    "get_active_window": ToolMeta(Destructiveness.READ, True, 0, "screen"),
    "wait_for_screen_text": ToolMeta(Destructiveness.READ, True, 0, "screen"),
    "zoom_screen": ToolMeta(Destructiveness.READ, True, 1, "screen"),
    "read_ui": ToolMeta(Destructiveness.READ, True, 0, "ui"),

    # ---------- UI control (WRITE — can be undone via state restore) ----------
    "click_screen": ToolMeta(Destructiveness.WRITE, False, 0, "ui"),
    "click_type": ToolMeta(Destructiveness.WRITE, False, 0, "ui"),
    "click_text_on_screen": ToolMeta(Destructiveness.WRITE, False, 0, "ui"),
    "click_zoom": ToolMeta(Destructiveness.WRITE, False, 0, "ui"),
    "type_text": ToolMeta(Destructiveness.WRITE, False, 0, "ui"),
    "press_hotkey": ToolMeta(Destructiveness.WRITE, False, 0, "ui"),
    "drag": ToolMeta(Destructiveness.WRITE, False, 0, "ui"),
    "draw_path": ToolMeta(Destructiveness.WRITE, False, 0, "ui"),
    "draw_shape": ToolMeta(Destructiveness.WRITE, False, 0, "ui"),
    "click_ui": ToolMeta(Destructiveness.WRITE, False, 0, "ui"),
    "set_field": ToolMeta(Destructiveness.WRITE, False, 0, "ui"),
    "wait_and_click": ToolMeta(Destructiveness.WRITE, False, 0, "ui"),
    "wait_and_press": ToolMeta(Destructiveness.WRITE, False, 0, "ui"),
    "skip_youtube_ad": ToolMeta(Destructiveness.WRITE, False, 0, "ui"),

    # ---------- windows / apps (WRITE) ----------
    "open_app": ToolMeta(Destructiveness.WRITE, False, 0, "system"),
    "open_in_editor": ToolMeta(Destructiveness.WRITE, False, 0, "system"),
    "open_url": ToolMeta(Destructiveness.WRITE, False, 0, "system"),
    "open_path": ToolMeta(Destructiveness.WRITE, False, 0, "files"),
    "control_window": ToolMeta(Destructiveness.WRITE, True, 0, "system"),
    "close_window": ToolMeta(Destructiveness.WRITE, False, 0, "system"),
    "move_window_to_monitor": ToolMeta(Destructiveness.WRITE, True, 0, "system"),

    # ---------- file system ----------
    "list_files": ToolMeta(Destructiveness.READ, True, 0, "files"),
    "list_recent_paths": ToolMeta(Destructiveness.READ, True, 0, "files"),
    "read_file": ToolMeta(Destructiveness.READ, True, 0, "files"),
    "create_folder": ToolMeta(Destructiveness.WRITE, True, 0, "files"),
    "create_file": ToolMeta(Destructiveness.WRITE, True, 0, "files"),
    "write_file": ToolMeta(Destructiveness.WRITE, True, 0, "files"),
    "append_file": ToolMeta(Destructiveness.WRITE, True, 0, "files"),
    "move_file": ToolMeta(Destructiveness.WRITE, True, 0, "files"),
    "rename_file": ToolMeta(Destructiveness.WRITE, True, 0, "files"),
    "delete_file": ToolMeta(Destructiveness.IRREVERSIBLE, False, 0, "files"),

    # ---------- scripts / agents ----------
    # run_quick_command is a deterministic dispatch into Touchless's
    # built-in command router (media / spotify / volume / pause /
    # next-song). Despite the "run" verb, it does NOT execute shell
    # — it's the natural-language front door for the same actions
    # the user can already trigger with gestures or hotkeys. Confirm
    # gate would friction-block "play poker face" with no benefit.
    "run_quick_command": ToolMeta(Destructiveness.WRITE, False, 0, "scripts"),
    "run_existing_touchless_action": ToolMeta(Destructiveness.WRITE, False, 0, "scripts"),
    "run_python_script": ToolMeta(Destructiveness.DESTRUCTIVE, False, 0, "scripts"),
    "run_matlab_script": ToolMeta(Destructiveness.DESTRUCTIVE, False, 0, "scripts"),
    "send_to_coding_agent": ToolMeta(Destructiveness.DESTRUCTIVE, False, 2, "agents"),
    "follow_up_coding_agent": ToolMeta(Destructiveness.DESTRUCTIVE, False, 2, "agents"),
    "auto_approve": ToolMeta(Destructiveness.DESTRUCTIVE, False, 0, "agents"),
    "stop_auto_approve": ToolMeta(Destructiveness.WRITE, True, 0, "agents"),
    "ask_user_confirmation": ToolMeta(Destructiveness.READ, True, 0, "interactive"),

    # ---------- web / browser ----------
    "web_search": ToolMeta(Destructiveness.READ, True, 1, "web"),
    "web_navigate": ToolMeta(Destructiveness.WRITE, False, 0, "web"),
    "web_get_links": ToolMeta(Destructiveness.READ, True, 0, "web"),
    "web_get_text": ToolMeta(Destructiveness.READ, True, 0, "web"),
    "web_click": ToolMeta(Destructiveness.WRITE, False, 0, "web"),
    "web_scroll": ToolMeta(Destructiveness.WRITE, True, 0, "web"),
    "web_fill": ToolMeta(Destructiveness.WRITE, False, 0, "web"),
    "web_wait_for": ToolMeta(Destructiveness.READ, True, 0, "web"),
    "web_eval": ToolMeta(Destructiveness.DESTRUCTIVE, False, 0, "web"),

    # ---------- ambient pack ----------
    "notify_toast": ToolMeta(Destructiveness.WRITE, True, 0, "notify"),
    "clipboard_read": ToolMeta(Destructiveness.READ, True, 0, "clipboard"),
    "clipboard_write": ToolMeta(Destructiveness.WRITE, True, 0, "clipboard"),
    "clipboard_transform": ToolMeta(Destructiveness.WRITE, True, 1, "clipboard"),

    # ---------- meta ----------
    "compose_text": ToolMeta(Destructiveness.READ, True, 0, "meta"),
    "weather_get": ToolMeta(Destructiveness.READ, True, 0, "external"),
    "find_capability": ToolMeta(Destructiveness.READ, True, 0, "meta"),
    "iris_add_project": ToolMeta(Destructiveness.WRITE, True, 0, "meta"),
    "iris_remove_project": ToolMeta(Destructiveness.WRITE, True, 0, "meta"),
    "iris_query_node": ToolMeta(Destructiveness.READ, True, 0, "meta"),
    "iris_describe_project": ToolMeta(Destructiveness.READ, True, 0, "meta"),

    # ---------- connectors: read-mostly ----------
    "volume_get": ToolMeta(Destructiveness.READ, True, 0, "audio"),
    "volume_list_apps": ToolMeta(Destructiveness.READ, True, 0, "audio"),
    "media_now_playing": ToolMeta(Destructiveness.READ, True, 0, "audio"),
    "ollama_list_models": ToolMeta(Destructiveness.READ, True, 0, "ai"),

    # ---------- connectors: write/local ----------
    "volume_set": ToolMeta(Destructiveness.WRITE, True, 0, "audio"),
    "volume_mute": ToolMeta(Destructiveness.WRITE, True, 0, "audio"),
    "volume_toggle_mute": ToolMeta(Destructiveness.WRITE, True, 0, "audio"),
    "volume_set_app": ToolMeta(Destructiveness.WRITE, True, 0, "audio"),
    "volume_mute_app": ToolMeta(Destructiveness.WRITE, True, 0, "audio"),
    "media_play_pause": ToolMeta(Destructiveness.WRITE, True, 0, "audio"),
    "media_next_track": ToolMeta(Destructiveness.WRITE, True, 0, "audio"),
    "media_previous_track": ToolMeta(Destructiveness.WRITE, True, 0, "audio"),
    "discord_mute": ToolMeta(Destructiveness.WRITE, True, 0, "comms"),
    "discord_deafen": ToolMeta(Destructiveness.WRITE, True, 0, "comms"),
    "discord_toggle_mute": ToolMeta(Destructiveness.WRITE, True, 0, "comms"),
    "discord_toggle_deafen": ToolMeta(Destructiveness.WRITE, True, 0, "comms"),
    "todo_add": ToolMeta(Destructiveness.WRITE, True, 0, "tasks"),
    "outlook_compose": ToolMeta(Destructiveness.WRITE, True, 0, "comms"),
    "ollama_generate": ToolMeta(Destructiveness.READ, True, 1, "ai"),

    # ---------- connectors: outbound / EXTERNAL VISIBILITY ----------
    # Anything that produces an outbound message / change visible to a
    # THIRD PARTY the user has not already granted access to is
    # DESTRUCTIVE. (Appending to the user's own doc is NOT — that's WRITE.)
    # Confirmation gate must catch these BEFORE invocation.
    "email_send": ToolMeta(Destructiveness.DESTRUCTIVE, False, 0, "comms"),
    "gmail_send": ToolMeta(Destructiveness.DESTRUCTIVE, False, 0, "comms"),
    "ms_mail_send": ToolMeta(Destructiveness.DESTRUCTIVE, False, 0, "comms"),
    "teams_send": ToolMeta(Destructiveness.DESTRUCTIVE, False, 0, "comms"),
    "teams_post": ToolMeta(Destructiveness.DESTRUCTIVE, False, 0, "comms"),
    "teams_channel_post": ToolMeta(Destructiveness.DESTRUCTIVE, False, 0, "comms"),
    "slack_post": ToolMeta(Destructiveness.DESTRUCTIVE, False, 0, "comms"),
    "slack_send": ToolMeta(Destructiveness.DESTRUCTIVE, False, 0, "comms"),
    "discord_send": ToolMeta(Destructiveness.DESTRUCTIVE, False, 0, "comms"),
    "phone_link_send_text": ToolMeta(Destructiveness.DESTRUCTIVE, False, 0, "comms"),

    # Calendar events visible to invitees.
    "calendar_create": ToolMeta(Destructiveness.DESTRUCTIVE, True, 0, "calendar"),
    "calendar_event_create": ToolMeta(Destructiveness.DESTRUCTIVE, True, 0, "calendar"),
    "ms_calendar_create": ToolMeta(Destructiveness.DESTRUCTIVE, True, 0, "calendar"),
    # Classic Outlook (COM) variant — same behavior + same risk profile
    # (invitees get pings) as the Graph and Google flavors. Was silently
    # falling to DEFAULT_META (write) and skipping the confirm.
    "outlook_com_create_event": ToolMeta(
        Destructiveness.DESTRUCTIVE, True, 0, "calendar"),

    # Cloud uploads — read USER FILES from disk and push to cloud. The
    # "what am I uploading" question warrants a confirm.
    "drive_upload": ToolMeta(Destructiveness.DESTRUCTIVE, True, 0, "cloud"),
    "onedrive_upload": ToolMeta(Destructiveness.DESTRUCTIVE, True, 0, "cloud"),
    # Creating a NEW EMPTY doc/sheet/slide/notebook is a WRITE, not
    # destructive: 1 new entity in the user's own Drive, zero external
    # side effects (nobody notified, nothing published), trivial undo
    # via right-click → Move to trash (30-day Drive trash retention).
    # Confirming empty-doc creation is friction with no safety benefit
    # and trains click-through fatigue for the confirms that matter.
    # Their populate/append counterparts are already WRITE — keep the
    # container consistent with the content.
    "gdocs_create": ToolMeta(Destructiveness.WRITE, True, 0, "cloud"),
    "gdocs_append_text": ToolMeta(Destructiveness.WRITE, True, 0, "cloud"),
    "sheets_create": ToolMeta(Destructiveness.WRITE, True, 0, "cloud"),
    # Row-append only ADDS new rows — no overwrite / delete of existing
    # data. Reversible by deleting the appended row, same reasoning that
    # keeps contacts_create at WRITE. Do not fire the confirm modal for a
    # safe append.
    "sheets_append_rows": ToolMeta(Destructiveness.WRITE, True, 0, "cloud"),
    # Pure read — values.get(). Safe to re-run indefinitely.
    "sheets_read_range": ToolMeta(Destructiveness.READ, True, 0, "cloud"),
    # Overwrite cells — same undo path as sheets_update_range; WRITE
    # because nothing leaves the account.
    "sheets_update_range": ToolMeta(Destructiveness.WRITE, True, 0, "cloud"),
    "sheets_clear_range": ToolMeta(Destructiveness.WRITE, True, 0, "cloud"),
    "slides_create": ToolMeta(Destructiveness.WRITE, True, 0, "cloud"),
    "slides_add_slide": ToolMeta(Destructiveness.WRITE, True, 0, "cloud"),
    "slides_replace_text": ToolMeta(Destructiveness.WRITE, True, 0, "cloud"),
    "onenote_create": ToolMeta(Destructiveness.WRITE, True, 0, "cloud"),
    "onenote_append_text": ToolMeta(Destructiveness.WRITE, True, 0, "cloud"),
    "excel_create": ToolMeta(Destructiveness.WRITE, True, 0, "cloud"),
    "word_create": ToolMeta(Destructiveness.WRITE, True, 0, "cloud"),
    "powerpoint_create": ToolMeta(Destructiveness.WRITE, True, 0, "cloud"),

    # Notion — page/database writes are visible to the user's
    # workspace + anyone they've shared with.
    "notion_create_page": ToolMeta(
        Destructiveness.DESTRUCTIVE, True, 0, "cloud"),
    "notion_append_to_page": ToolMeta(
        Destructiveness.WRITE, True, 0, "cloud"),
    "notion_add_to_database": ToolMeta(
        Destructiveness.WRITE, True, 0, "cloud"),
    "notion_search": ToolMeta(Destructiveness.READ, True, 0, "cloud"),

    # Calendar — explicit create_event variant the Google connector uses.
    "calendar_create_event": ToolMeta(
        Destructiveness.DESTRUCTIVE, True, 0, "calendar"),

    # Spotify playback — write to the user's audio output, easy
    # to undo with another command. Not "destructive" enough to
    # require a confirm; just write.
    "spotify_play": ToolMeta(Destructiveness.WRITE, True, 0, "audio"),
    "spotify_pause": ToolMeta(Destructiveness.WRITE, True, 0, "audio"),

    # Read-only listing of cloud items — safe.
    "drive_list": ToolMeta(Destructiveness.READ, True, 0, "cloud"),
    "ms_mail_list": ToolMeta(Destructiveness.READ, True, 0, "comms"),
    "ms_mail_search": ToolMeta(Destructiveness.READ, True, 0, "comms"),
    "calendar_list_events": ToolMeta(Destructiveness.READ, True, 0, "calendar"),
    "gmail_list": ToolMeta(Destructiveness.READ, True, 0, "comms"),
    "gmail_read": ToolMeta(Destructiveness.READ, True, 0, "comms"),
    "contacts_search": ToolMeta(Destructiveness.READ, True, 0, "comms"),
    "contacts_list": ToolMeta(Destructiveness.READ, True, 0, "comms"),
    "outlook_com_contacts_search": ToolMeta(
        Destructiveness.READ, True, 0, "comms"),
    # contacts_create writes into the user's address book — local-only
    # but persistent. WRITE (reversible via Google Contacts UI), not
    # DESTRUCTIVE since nothing leaves the account.
    "contacts_create": ToolMeta(Destructiveness.WRITE, True, 0, "comms"),

    # Google Tasks — local-only task list mutations. tasks_list / read
    # is harmless; add is a reversible WRITE; complete flips a status
    # bit (reversible WRITE); delete is permanent IRREVERSIBLE since
    # the API has no trash.
    "tasks_list": ToolMeta(Destructiveness.READ, True, 0, "tasks"),
    "tasks_add": ToolMeta(Destructiveness.WRITE, True, 0, "tasks"),
    "tasks_complete": ToolMeta(Destructiveness.WRITE, True, 0, "tasks"),
    "tasks_delete": ToolMeta(Destructiveness.IRREVERSIBLE, False, 0, "tasks"),

    # Google Forms — creating a poll/survey produces a sharable URL
    # (visible to anyone with the link), so forms_create is DESTRUCTIVE
    # in the same sense as gdocs_create (cloud artifact, externally
    # visible). forms_responses is a pure read.
    "forms_create": ToolMeta(Destructiveness.DESTRUCTIVE, True, 0, "cloud"),
    "forms_responses": ToolMeta(Destructiveness.READ, True, 0, "cloud"),

    # YouTube Data API readonly — list playlists / items / subscriptions.
    "youtube_my_playlists": ToolMeta(Destructiveness.READ, True, 0, "media"),
    "youtube_playlist_items": ToolMeta(Destructiveness.READ, True, 0, "media"),
    "youtube_subscriptions": ToolMeta(Destructiveness.READ, True, 0, "media"),

    # Google Photos appendonly — uploads a local file to the user's
    # cloud library. Visible to anyone the user shares the album / link
    # with, so DESTRUCTIVE matches drive_upload / onedrive_upload.
    "photos_upload": ToolMeta(Destructiveness.DESTRUCTIVE, True, 0, "cloud"),

    # Identity / profile reads — entirely server-side reads of the
    # user's own account. No side effects.
    "google_whoami": ToolMeta(Destructiveness.READ, True, 0, "memory"),
    "google_my_birthday": ToolMeta(Destructiveness.READ, True, 0, "memory"),

    # IRREVERSIBLE — delete / pay.
    "file_delete": ToolMeta(Destructiveness.IRREVERSIBLE, False, 0, "files"),
    "drive_delete": ToolMeta(Destructiveness.IRREVERSIBLE, False, 0, "cloud"),
    "mail_delete": ToolMeta(Destructiveness.IRREVERSIBLE, False, 0, "comms"),

    # ---------- iris pseudo-tools (Tier-1 classifier targets) ----------
    "iris_lookup_contact": ToolMeta(Destructiveness.READ, True, 0, "memory"),
    "iris_remember_contact": ToolMeta(Destructiveness.WRITE, True, 0, "memory"),
    "iris_forget_contact": ToolMeta(Destructiveness.IRREVERSIBLE, False, 0, "memory"),
    "iris_set_preference": ToolMeta(Destructiveness.WRITE, True, 0, "memory"),
    "iris_open_last": ToolMeta(Destructiveness.WRITE, False, 0, "system"),
    "iris_setup_tool": ToolMeta(Destructiveness.WRITE, True, 0, "setup"),
}


# ---- lookups --------------------------------------------------------------


def get_metadata(tool_name: str) -> ToolMeta:
    """Return the metadata for `tool_name`, or DEFAULT_META if unknown.
    Conservative default ensures unregistered tools route through the
    confirm gate instead of slipping through as if read-only."""
    return DEFAULT_METADATA.get(tool_name, DEFAULT_META)


def destructiveness_of(tool_name: str) -> Destructiveness:
    return get_metadata(tool_name).destructiveness


def is_reversible(tool_name: str) -> bool:
    return get_metadata(tool_name).reversible


def is_destructive_or_worse(tool_name: str) -> bool:
    """True for DESTRUCTIVE or IRREVERSIBLE. Useful gate predicate —
    the confirm gate + send gate use this exact set."""
    d = destructiveness_of(tool_name)
    return d in (Destructiveness.DESTRUCTIVE, Destructiveness.IRREVERSIBLE)


def register_metadata(tool_name: str, meta: ToolMeta) -> None:
    """Allow connectors / MCP bridges to register metadata at startup.
    Used by the MCP bridge to declare tier-specific metadata once a
    server's tool list is enumerated."""
    DEFAULT_METADATA[tool_name] = meta


def category_of(tool_name: str) -> str:
    return get_metadata(tool_name).category or "unknown"
