"""Capability connectors for iris (OpenClaw-style, native Python).

A connector bundles, for one capability/service: the tool schemas it
exposes to the Realtime model, whether it's usable right now, and the
handler that runs a tool. The registry assembles available connectors'
tools into the model's tool list and routes calls to the owning
connector — the fast, deterministic API-first path. Anything no
connector handles falls through to the GUI computer-use executor (the
universal fallback), so iris reaches both API-backed services AND any
app without an API.

Author: Konstantin Markov
"""
from typing import Any, Optional

from .base import Connector, ConnectorRegistry, connector_result

__all__ = [
    "Connector",
    "ConnectorRegistry",
    "connector_result",
    "build_connector_registry",
]


def build_connector_registry(executor: Optional[Any] = None) -> ConnectorRegistry:
    """Assemble the default set of OpenClaw-style connectors.

    Each connector is registered unconditionally; whether its tools reach
    the model is decided later, per session, by `available()` (configured /
    authorized / a relevant window open). A connector that fails to import
    is skipped so one broken connector never sinks the rest.

    `executor` is the session's ToolExecutor; connectors for services the
    executor already owns (YouTube, Chrome) reuse its single instance.
    """
    reg = ConnectorRegistry()

    def _add(factory) -> None:
        try:
            reg.register(factory())
        except Exception:
            # Import/construction failure (e.g. optional dep missing) must
            # not take down the whole registry.
            pass

    # NOTE: SpotifyConnector exists (spotify_connector.py) but is deliberately
    # NOT registered. Spotify is owned by the Touchless Layer-0 command router
    # (play/pause/next/search), which handles it before the model. A connector
    # was a redundant second path that mis-parsed requests (e.g. "play feel
    # good rock playlist" played a single track), so Spotify is left to
    # Touchless. Re-register here only if Layer 0 stops covering Spotify.
    from .volume_connector import VolumeConnector
    from .media_connector import MediaConnector
    from .youtube_connector import YouTubeConnector
    from .chrome_connector import ChromeConnector
    from .discord_connector import DiscordConnector
    from .outlook_connector import OutlookConnector
    from .office_connector import OfficeConnector
    from .gmail_connector import GmailConnector
    from .calendar_connector import CalendarConnector
    from .gdocs_connector import GoogleDocsConnector
    from .sheets_connector import GoogleSheetsConnector
    from .slides_connector import GoogleSlidesConnector
    from .drive_connector import DriveConnector
    from .directions_connector import DirectionsConnector
    from .ms365_connector import Microsoft365Connector
    from .kicad_cli_connector import KiCadCliConnector

    _add(lambda: VolumeConnector())
    _add(lambda: MediaConnector())
    _add(lambda: YouTubeConnector(executor=executor))
    _add(lambda: ChromeConnector(executor=executor))
    _add(lambda: DiscordConnector())
    _add(lambda: OutlookConnector(executor=executor))
    _add(lambda: OfficeConnector())
    _add(lambda: GmailConnector())
    _add(lambda: CalendarConnector())
    _add(lambda: GoogleDocsConnector())
    _add(lambda: GoogleSheetsConnector())
    _add(lambda: GoogleSlidesConnector())
    _add(lambda: DriveConnector())
    _add(lambda: DirectionsConnector())
    _add(lambda: Microsoft365Connector())
    _add(lambda: KiCadCliConnector())

    # MCP servers (breadth for everything not hand-written). Each configured
    # server becomes a connector whose tools the search router can discover.
    # Empty unless the `mcp` sdk + a server config are present.
    try:
        from .mcp_bridge import build_mcp_connectors
        for mc in build_mcp_connectors():
            reg.register(mc)
    except Exception:
        pass

    # Synonym-rich descriptions power the capability-search router's
    # matching (e.g. "play some music" -> spotify, "schedule a meeting"
    # -> calendar). Kept here so all the intent vocabulary lives in one
    # place rather than scattered across connector files.
    descriptions = {
        "youtube": "YouTube videos in the browser: play pause seek speed captions fullscreen like search",
        "discord": "Discord voice chat: mute deafen join leave voice channel",
        "volume": "system audio sound volume level mute unmute speakers",
        "media": "global media play pause key for whatever is currently playing",
        "chrome": "Google Chrome web browser: back forward refresh new tab search open url website",
        "outlook": "email mail compose send write message Outlook inbox folder",
        "office": "Microsoft Office documents Word Excel PowerPoint spreadsheet presentation slide save pdf",
        "gmail": "Gmail email mail send compose write message",
        "calendar": "Google Calendar schedule meeting appointment event agenda reminder",
        "gdocs": "Google Docs document create write notes",
        "gsheets": "Google Sheets spreadsheet create table rows data",
        "gslides": "Google Slides presentation slideshow deck create",
        "drive": "Google Drive upload save file list cloud storage",
        "directions": "directions route navigation distance drive travel time map between places",
        "ms365": "Microsoft 365 Outlook email send read search Microsoft calendar event OneDrive upload Teams message chat Excel spreadsheet cell To Do task reminder OneNote note Contacts Office Copilot",
    }
    for c in reg._connectors:
        if not getattr(c, "description", ""):
            c.description = descriptions.get(getattr(c, "id", ""), "")
    return reg
