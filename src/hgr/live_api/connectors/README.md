# iris connectors (OpenClaw-style, API-first)

A connector gives iris a **fast, deterministic API path** for a service so
the Realtime model doesn't have to drive the app's UI by screenshot +
click. Each connector declares the tool schemas it exposes, whether it's
usable right now (`available()`), and how to run a tool (`execute()`).

The `ConnectorRegistry` merges only the **available** connectors' tools
into the model's tool list each session, and routes a call to its owner.
Anything no connector handles falls through to the GUI computer-use
executor — the universal fallback that reaches *any* app. So:

- **Connectors** = the fast lane for a curated set of services.
- **Computer-use** (`click_screen`, `type_text`, `click_ui`, `web_*`) =
  the universal road that already controls every other app.

> You do **not** need a connector to "use an app" — computer-use already
> can. A connector is a *speed/reliability upgrade* for high-frequency,
> API-backed services.

## Connectors in this build

| Connector | Tools | Available when… | Controller source |
|---|---|---|---|
| `spotify` | play/pause/next/prev/volume/now-playing/search-play/add-to-playlist | Spotify authorized | `debug/spotify_controller` |
| `youtube` | play/pause/seek/speed/volume/captions/fullscreen/like/search-play | a YouTube tab is open | `debug/youtube_controller` (shared) |
| `discord` | mute/deafen (+toggle)/voice-status/join/leave voice | Discord authorized | `debug/discord_controller` |
| `volume` | get/set/mute/toggle system volume | always (Windows) | `debug/volume_controller` |
| `media` | global media play/pause | always (Windows) | `debug/media_controller` |
| `chrome` | back/forward/refresh/new-tab/search/open-url | Chrome is running | `debug/chrome_controller` (shared) |
| `outlook` | compose/open/open-folder | always (mailto) | `debug/desktop_controller` (shared) |
| `office` | Word/Excel/PowerPoint create/edit/save-as-PDF | that Office app is installed | `debug/office_controller` (COM) |
| `gmail` | send/list-unread/search | Google authorized | Gmail API |
| `calendar` | list/create events | Google authorized | Google Calendar API |
| `gdocs` | create doc | Google authorized | Google Docs API |

Availability gating is the efficiency mechanism: tools the model never
sees can't bloat the prompt or be mis-picked.

## Office (Word / Excel / PowerPoint)

Uses **comtypes** COM automation (no pywin32). Per-app gating reads the
ProgID from the registry, so on a machine with only Word the Excel/PPT
tools never appear. Apps launch visible so you review before saving;
nothing auto-sends/prints. OneNote is intentionally omitted (its COM model
is XML-based and brittle — that belongs on the Graph API path later).

> Office paths are implemented but require an Office-installed machine to
> verify end-to-end (none was available where this was built).

## Google (Gmail / Calendar / Docs) — one-click for users

End users do **one thing**: click **"Connect Gmail"** in the assistant
window → approve Google's consent screen → done (token saved locally). They
never create a Google Cloud project.

That works because the app embeds **your** (the developer's) OAuth client.
Setup split:

**Developer, once:**
1. `pip install google-api-python-client google-auth google-auth-oauthlib`
   (ship these in the build).
2. In [Google Cloud Console](https://console.cloud.google.com/): create a
   project, enable Gmail/Calendar/Docs/Drive APIs, create **OAuth client ID →
   Desktop app**.
3. Embed it — set env (or bundle): `GOOGLE_OAUTH_CLIENT_ID` +
   `GOOGLE_OAUTH_CLIENT_SECRET` (a Desktop client secret is a *public-client*
   secret + PKCE, so shipping it is per Google's installed-app model). A
   bundled JSON via `GOOGLE_OAUTH_CLIENT_JSON` also works.
4. **For public release:** submit the app for Google **OAuth verification**
   (Gmail scopes are sensitive/restricted). Until verified, only test users
   you add can connect, with an "unverified app" warning. *This is the one
   unavoidable gate — inherent to Gmail scopes, not this code.*

**User, once:** click **Connect Gmail** → Allow. (CLI equivalent for dev:
`python -m hgr.live_api.connectors.google_client authorize`.)

Mechanics: `google_client.status()` drives the button
(`needs_libs`/`needs_client`/`ready_to_connect`/`connected`); `connect()`
runs `InstalledAppFlow.run_local_server()` on a worker thread and saves the
token; it refreshes silently after. Scopes: gmail.send, calendar,
drive.file — **all "sensitive" (free verification, no paid CASA
assessment)**. `gmail.readonly` is intentionally excluded (it's
"restricted" and would require the paid assessment for public release), so
Gmail is **send-only**. Sending (`gmail_send`) is **confirmed** in the UI
before it fires.

**Docs / Sheets / Slides — drive.file migration (2026-07-29).** The
broad `documents` / `spreadsheets` / `presentations` scopes were removed
after Google's OAuth verification review rejected them (restricted-tier).
Docs v1, Sheets v4, and Slides v1 all accept `drive.file`-issued tokens
as long as the API caller has a `file_id` for a file that was either
CREATED by this app (all `*_connector.create` paths satisfy this
automatically — the freshly-minted file is warm-cached under its title
for immediate append-by-name follow-ups) or explicitly opened by the
user via the Google Picker widget (`app/ui/google_picker_dialog.py`).
Picked ids are remembered in `google_picker_cache.PickerCache` keyed
by slug so each file only needs to be picked once. See
`google_picker_cache.py` for the SQLite schema and slugify rules.

## Capability-search router (how tools stay lean)

Connector tools are **not** all dumped into the model's context — that
would bloat the prompt and hurt tool-selection accuracy (the opposite of
the efficiency we want). Instead the session starts with a lean list:

```
all built-in tools  +  one meta-tool: find_capability
```

When the model wants to act on an app, it calls `find_capability("play
music")`. The manager (`_handle_find_capability`) searches the registry
(`search()` ranks available connectors by word overlap with the task),
loads the winning connector's tool schemas into the live session via
`client.update_tools(...)` (a `session.update` re-push), and tells the
model which tool to call. The model then calls e.g. `spotify_play`.

So the model only ever holds a handful of tools, but can reach *any*
connector on demand — the same trick brokers use to scale to hundreds of
integrations. Dispatch routes through `ToolRegistry.call()`, so a loaded
connector tool runs on its connector and everything else falls through to
the executor.

Key pieces: `ConnectorRegistry.search()/catalog()` (base.py),
`FIND_CAPABILITY_TOOL` + `_handle_find_capability` + `_current_tool_schemas`
(live_api_manager.py), `update_tools()` (realtime_client.py / local_backend.py).

## MCP bridge (breadth without hand-writing connectors)

`mcp_bridge.py` turns any **MCP (Model Context Protocol) server** into
connectors — one server ≈ many tools/apps, maintained by the MCP ecosystem
instead of by us. Each configured server is wrapped as an `MCPConnector`
(tools namespaced `mcp_<server>_<tool>`) and registered into the same
registry, so `find_capability` discovers MCP tools alongside the
hand-written ones. Servers run as local child processes over stdio
(self-hosted, data stays on the machine — no cloud broker).

Ships **dormant**. To enable:

1. `pip install mcp`
2. Create `~/Documents/Touchless/mcp_servers.json` (override with
   `TOUCHLESS_MCP_CONFIG`):
   ```json
   {
     "servers": [
       {"name": "filesystem", "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-filesystem", "C:\\Users\\me"]}
     ]
   }
   ```

The bridge marshals the MCP SDK's async calls onto a per-server background
event loop so the sync connector `execute()` can drive them.

> Strategy: **hand-write** connectors for high-value local/deep services
> (depth), **bridge MCP** for everything else (breadth). The GUI
> computer-use fallback still covers anything neither reaches.

## Adding a connector

1. Subclass `Connector` (see `spotify_connector.py` as the template).
2. Implement `tools()`, `available()`, `execute()`.
3. Register it in `build_connector_registry()` in `__init__.py`.
4. Prefix tool names with the connector id to avoid colliding with the
   built-in schemas in `../schemas.py`.

Author: Konstantin Markov
