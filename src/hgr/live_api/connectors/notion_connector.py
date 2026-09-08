"""Notion connector — read/write the user's Notion workspace via its REST API.

Lets Iris use Notion as ambient storage: search pages, read content, append
to pages, create pages, add rows to databases — all without ever opening the
Notion UI. Notes capture and recall stop being a destination the user goes
to and become something Iris does on their behalf in one step.

`setup_self()` handles the user-facing onboarding: opens the Notion
integrations page in their browser, tells them exactly what to create and
where to paste the token, and validates the token against /v1/users/me.

Notion's permission model is opt-in: an integration can ONLY see pages and
databases the user has explicitly shared with it (Share -> Add connections
on each page). setup_self() spells this out so first use isn't "why doesn't
search return anything."

No third-party SDK — Notion's REST API is small enough that stdlib urllib
keeps the surface honest and the deps minimal.

Author: Konstantin Markov
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
import webbrowser
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .base import Connector, connector_result


# Notion REST API. Version pinned so an upstream contract change can't
# silently corrupt our requests — bump deliberately when Notion ships one.
NOTION_API_BASE = "https://api.notion.com/v1"
NOTION_API_VERSION = "2022-06-28"
# Where the user pastes the integration token. Same per-app subdir pattern
# the Google/MCP connectors use under ~/Documents/Touchless/.
TOKEN_DIR_RELATIVE = ("Documents", "Touchless", "notion")
TOKEN_FILE_NAME = "notion_token.txt"
# Page Notion shows for creating an internal integration. We open it as part
# of setup_self so the user doesn't have to hunt for the URL.
NOTION_INTEGRATIONS_URL = "https://www.notion.so/profile/integrations"
# HTTP timeouts kept small — every call is a couple of KB; if Notion is slow
# we'd rather surface an error than hang the realtime turn.
HTTP_TIMEOUT_DEFAULT = 12.0


def _token_path() -> Path:
    """Absolute path to the token file. ~/Documents/Touchless/notion/notion_token.txt."""
    home = Path.home()
    return home.joinpath(*TOKEN_DIR_RELATIVE) / TOKEN_FILE_NAME


def _load_token() -> str:
    """Read the saved Notion integration token. Empty string if missing or
    unreadable — caller treats both as 'not set up yet'."""
    path = _token_path()
    try:
        if not path.exists():
            return ""
        return path.read_text(encoding="utf-8").strip()
    except Exception:
        return ""


def _save_token(token: str) -> Path:
    """Write the token to disk, creating the parent dir 0o700. Returns the
    path so setup_self can tell the user where it landed."""
    path = _token_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    # Best-effort: tighten perms on POSIX so other users on a shared box can't
    # read the token. No-op on Windows (NTFS ACLs handled elsewhere).
    try:
        os.chmod(path.parent, 0o700)
    except Exception:
        pass
    path.write_text(token.strip() + "\n", encoding="utf-8")
    return path


def _request(method: str, path: str, token: str,
             body: Optional[Dict[str, Any]] = None,
             timeout: float = HTTP_TIMEOUT_DEFAULT
             ) -> Tuple[int, Dict[str, Any]]:
    """Make a Notion API call. Returns (http_status, parsed_json). Never
    raises — network/JSON errors come back as (0, {'error': ...}) so the
    connector can produce a useful result instead of a stack trace."""
    headers = {
        "Authorization": f"Bearer {token}",
        "Notion-Version": NOTION_API_VERSION,
        "Accept": "application/json",
    }
    data: Optional[bytes] = None
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(
        NOTION_API_BASE + path, data=data, headers=headers, method=method
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            return (resp.status, json.loads(raw) if raw else {})
    except urllib.error.HTTPError as exc:
        # Notion sends a JSON error body — surface it instead of just the
        # status code so the model can tell the user something useful.
        try:
            err_body = exc.read().decode("utf-8", errors="replace")
            return (exc.code, json.loads(err_body) if err_body else
                    {"error": str(exc)})
        except Exception:
            return (exc.code, {"error": str(exc)})
    except Exception as exc:
        return (0, {"error": f"{type(exc).__name__}: {exc}"})


# ---- block helpers ---------------------------------------------------------
# Notion stores page content as "blocks." The API reads them in chunks and
# writes them as JSON. These helpers convert between Notion's structured
# format and the flat text the model actually wants to work with.

def _paragraph_block(text: str) -> Dict[str, Any]:
    """Build a 'paragraph' block from a string. Notion caps each rich-text
    run at 2000 chars; we split longer text into multiple runs in one block
    so the model can dump a multi-line summary without thinking about it."""
    # Notion limits each rich_text item to 2000 chars; one block can hold
    # multiple runs concatenated visually, so we chunk on that boundary.
    runs: List[Dict[str, Any]] = []
    s = str(text or "")
    while s:
        chunk, s = s[:1900], s[1900:]
        runs.append({"type": "text", "text": {"content": chunk}})
    if not runs:
        runs = [{"type": "text", "text": {"content": ""}}]
    return {
        "object": "block",
        "type": "paragraph",
        "paragraph": {"rich_text": runs},
    }


def _blocks_from_text(text: str) -> List[Dict[str, Any]]:
    """Turn a multi-line string into a sequence of paragraph blocks (one per
    line). Empty lines map to empty paragraphs, preserving the user's spacing.
    Notion caps a single append at 100 blocks; the caller handles batching."""
    lines = str(text or "").splitlines() or [""]
    return [_paragraph_block(line) for line in lines]


def _plain_text_from_block(block: Dict[str, Any]) -> str:
    """Best-effort flatten of a Notion block into one line of plain text.
    Handles the common content types — paragraph, headings, list items,
    to_do, code, quote, callout. Unrecognised types fall through to empty so
    we never raise on a content type that's new since this was written."""
    btype = str(block.get("type") or "")
    body = block.get(btype) or {}
    rich = body.get("rich_text") or []
    text = "".join(str(rt.get("plain_text") or "") for rt in rich)
    if btype == "paragraph":
        return text
    if btype.startswith("heading_"):
        # heading_1/2/3 -> # / ## / ###
        try:
            level = int(btype[-1])
        except Exception:
            level = 1
        return f"{'#' * max(1, min(3, level))} {text}"
    if btype == "bulleted_list_item":
        return f"- {text}"
    if btype == "numbered_list_item":
        return f"1. {text}"  # markdown re-numbers on render
    if btype == "to_do":
        checked = bool(body.get("checked"))
        return f"[{'x' if checked else ' '}] {text}"
    if btype == "code":
        lang = body.get("language") or ""
        return f"```{lang}\n{text}\n```"
    if btype == "quote":
        return f"> {text}"
    if btype == "callout":
        return f"💡 {text}"
    if btype == "child_page":
        return f"📄 {body.get('title', '')}"
    if btype == "divider":
        return "---"
    return text  # unknown type — return whatever rich_text we found


class NotionConnector(Connector):
    """Notion REST API — search, read, write pages and databases."""

    id = "notion"
    description = (
        "Notion workspace: search pages and databases, read page content, "
        "append notes to a page, create new pages, add rows to a tracker "
        "database. Use for note capture, journal writes, task creation, "
        "and recall — without opening Notion's UI."
    )

    def __init__(self, *, setup_only: bool = False) -> None:
        # Token loaded eagerly so the first available() check is free. We
        # re-load lazily on auth failures (in case the user rotated it).
        self._token: str = _load_token()
        # setup_only=True hides tools() but keeps setup_self() reachable, so
        # 'set up notion' / iris_setup_tool('notion') still work. Used to
        # demote the connector when the user doesn't write into Notion — its
        # tools then don't compete with web_search for 'my X' phrases.
        self._setup_only = bool(setup_only)

    # ---- setup_self ---------------------------------------------------------

    def setup_self(self, token: str = "") -> Dict[str, Any]:
        """Configure the Notion integration. Three branches:

        (1) `token` arg provided → save it, validate, done.
        (2) Token file already exists and is valid → idempotent ok.
        (3) Otherwise → open the Notion integrations page in the browser
            and return precise instructions for what the user needs to do.
        """
        # Branch 1: explicit token (e.g. 'set up notion with token ntn_xxx').
        explicit = (token or "").strip()
        if explicit:
            saved_at = _save_token(explicit)
            self._token = explicit
            ok, who = self._validate_token(explicit)
            if not ok:
                return {
                    "ok": False,
                    "error": ("Saved the token but Notion rejected it. "
                              "Double-check you copied the WHOLE 'Internal "
                              "Integration Secret' (starts with 'ntn_' or "
                              "'secret_') from the integration's settings "
                              "page, not a workspace URL."),
                    "token_path": str(saved_at),
                }
            return {
                "ok": True,
                "integration": who,
                "token_path": str(saved_at),
                "message": ("Notion is connected. IMPORTANT: open each page "
                            "or database you want me to read/write, click "
                            "Share -> Add connections, and add the Iris "
                            "integration — Notion's permission model is "
                            "opt-in per page."),
            }

        # Branch 2: existing valid token.
        if self._token:
            ok, who = self._validate_token(self._token)
            if ok:
                return {
                    "ok": True,
                    "integration": who,
                    "already_authorized": True,
                    "message": ("Notion is already connected. If search "
                                "isn't finding a page, make sure that page "
                                "has the Iris integration added under "
                                "Share -> Add connections."),
                }
            # Token present but rejected — fall through to re-onboarding.

        # Branch 3: not set up. Open the Notion integrations page in the
        # browser and tell the user exactly what to do next.
        try:
            webbrowser.open(NOTION_INTEGRATIONS_URL)
        except Exception:
            pass
        path = _token_path()
        return {
            "ok": False,
            "next_action": "user_must_create_integration_and_save_token",
            "error": (
                "Notion needs a one-time integration setup (~1 minute). "
                "I've opened https://www.notion.so/profile/integrations in "
                "your browser. There:\n"
                "  1. Click '+ New integration' (or 'Develop your own "
                "integrations').\n"
                "  2. Name it 'Iris' (or anything), pick your workspace.\n"
                "  3. After it's created, copy the 'Internal Integration "
                "Secret' (starts with 'ntn_' or 'secret_').\n"
                f"  4. Save that token to: {path}\n"
                "  5. Open each page/database you want me to access and "
                "click Share -> Add connections -> Iris.\n"
                "  6. Then say 'set up notion' again."
            ),
            "token_path": str(path),
            "integrations_url": NOTION_INTEGRATIONS_URL,
        }

    def _validate_token(self, token: str) -> Tuple[bool, Optional[str]]:
        """Hit /v1/users/me to check the token works. Returns (ok, label) —
        label is the integration's bot name if Notion returns one."""
        status, body = _request("GET", "/users/me", token, timeout=8.0)
        if status != 200:
            return False, None
        name = (body.get("bot") or {}).get("workspace_name") or body.get("name")
        return True, name

    # ---- registry hooks -----------------------------------------------------

    def available(self) -> bool:
        # Cheap availability: we have a token. We deliberately do NOT hit the
        # network on every available() call — that would slow the tool-list
        # assembly. A bad token surfaces as a routed-call error instead.
        if self._token:
            return True
        # Refresh from disk in case the user just saved it (no restart needed).
        self._token = _load_token()
        return bool(self._token)

    def tools(self) -> List[Dict[str, Any]]:
        if self._setup_only:
            return []
        def fn(name: str, desc: str,
               props: Optional[Dict[str, Any]] = None,
               required: Optional[List[str]] = None) -> Dict[str, Any]:
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
            fn(
                "notion_search",
                "Search the user's Notion workspace for pages and databases. "
                "Returns matches the integration has access to — if a page "
                "isn't found, the user may not have shared it with the Iris "
                "integration yet (Share -> Add connections on that page).",
                {
                    "query": {
                        "type": "string",
                        "description": "Free-text search query.",
                    },
                    "filter": {
                        "type": "string",
                        "enum": ["page", "database", "any"],
                        "default": "any",
                        "description": "Limit to pages, databases, or both.",
                    },
                    "limit": {
                        "type": "integer",
                        "default": 10,
                        "description": "Max results (1-50).",
                    },
                },
                ["query"],
            ),
            fn(
                "notion_read_page",
                "Read a Notion page's content as plain text. Returns the "
                "page title plus the text of every block (paragraphs, "
                "headings, lists, to-dos, code, quotes). Use to recall "
                "what's in a note. Pass the page_id from notion_search.",
                {
                    "page_id": {"type": "string"},
                    "max_blocks": {
                        "type": "integer",
                        "default": 200,
                        "description": "Stop after this many blocks (caps "
                                       "very long pages).",
                    },
                },
                ["page_id"],
            ),
            fn(
                "notion_append_to_page",
                "Append text to an existing Notion page. Each newline becomes "
                "its own paragraph block. Use for journal entries, quick "
                "notes, daily logs, dumping conversation summaries.",
                {
                    "page_id": {"type": "string"},
                    "text": {
                        "type": "string",
                        "description": "Multi-line text to append.",
                    },
                },
                ["page_id", "text"],
            ),
            fn(
                "notion_create_page",
                "Create a new Notion page under a parent page. Title is "
                "required; optional body text becomes paragraph blocks. "
                "Pass parent_page_id (NOT a database — use "
                "notion_add_to_database for those).",
                {
                    "parent_page_id": {"type": "string"},
                    "title": {"type": "string"},
                    "body": {
                        "type": "string",
                        "description": "Optional multi-line body text.",
                    },
                },
                ["parent_page_id", "title"],
            ),
            fn(
                "notion_add_to_database",
                "Add a new row to a Notion database (e.g. a task tracker). "
                "`title` fills the database's title column. `properties` is "
                "a mapping of OTHER property names to simple values (string "
                "for select/text, number for number, list for multi_select, "
                "ISO date string for date). The connector wraps them in "
                "Notion's expected schema. To know the property names, "
                "call notion_search to find the database first.",
                {
                    "database_id": {"type": "string"},
                    "title": {"type": "string"},
                    "properties": {
                        "type": "object",
                        "description": "Other property values keyed by "
                                       "property NAME.",
                    },
                },
                ["database_id", "title"],
            ),
        ]

    # ---- execution ----------------------------------------------------------

    def execute(self, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        # Re-load the token from disk if the in-memory copy was cleared (or
        # if the user just pasted it post-startup and never said 'set up').
        if not self._token:
            self._token = _load_token()
        if not self._token:
            return connector_result(
                "error",
                error=("Notion isn't set up. Say 'set up notion' and I'll "
                       "walk you through it."),
                code="not_configured",
            )

        if name == "notion_search":
            return self._do_search(args)
        if name == "notion_read_page":
            return self._do_read_page(args)
        if name == "notion_append_to_page":
            return self._do_append(args)
        if name == "notion_create_page":
            return self._do_create_page(args)
        if name == "notion_add_to_database":
            return self._do_add_to_database(args)

        return connector_result(
            "error",
            error=f"unknown notion tool: {name}",
            code="no_handler",
        )

    # ---- tool implementations ----------------------------------------------

    def _do_search(self, args: Dict[str, Any]) -> Dict[str, Any]:
        query = str(args.get("query") or "").strip()
        if not query:
            return connector_result("error", error="query is required",
                                    code="invalid_arguments")
        filt = str(args.get("filter") or "any").strip().lower()
        limit = max(1, min(50, int(args.get("limit") or 10)))
        body: Dict[str, Any] = {"query": query, "page_size": limit}
        if filt in ("page", "database"):
            body["filter"] = {"value": filt, "property": "object"}
        status, data = _request("POST", "/search", self._token, body=body)
        if status != 200:
            return self._http_error("search failed", status, data)
        results: List[Dict[str, Any]] = []
        for r in (data.get("results") or [])[:limit]:
            results.append({
                "id": r.get("id", ""),
                "object": r.get("object", ""),
                "title": _result_title(r),
                "url": r.get("url", ""),
                "last_edited": r.get("last_edited_time", ""),
            })
        return connector_result("ok", count=len(results), results=results)

    def _do_read_page(self, args: Dict[str, Any]) -> Dict[str, Any]:
        page_id = str(args.get("page_id") or "").strip()
        if not page_id:
            return connector_result("error", error="page_id is required",
                                    code="invalid_arguments")
        max_blocks = max(1, min(500, int(args.get("max_blocks") or 200)))
        # Page meta (mostly for the title).
        status, meta = _request("GET", f"/pages/{page_id}", self._token)
        if status != 200:
            return self._http_error("page fetch failed", status, meta)
        title = _result_title(meta)
        # Page content. Notion paginates blocks; walk pages until we hit
        # max_blocks or run out.
        lines: List[str] = []
        cursor = ""
        collected = 0
        while collected < max_blocks:
            path = f"/blocks/{page_id}/children?page_size=100"
            if cursor:
                path += f"&start_cursor={cursor}"
            status, body = _request("GET", path, self._token)
            if status != 200:
                return self._http_error("block fetch failed", status, body)
            for block in (body.get("results") or []):
                lines.append(_plain_text_from_block(block))
                collected += 1
                if collected >= max_blocks:
                    break
            if not body.get("has_more"):
                break
            cursor = str(body.get("next_cursor") or "")
            if not cursor:
                break
        return connector_result(
            "ok",
            title=title,
            url=meta.get("url", ""),
            block_count=collected,
            text="\n".join(lines),
        )

    def _do_append(self, args: Dict[str, Any]) -> Dict[str, Any]:
        page_id = str(args.get("page_id") or "").strip()
        text = args.get("text")
        if not page_id or not text:
            return connector_result("error",
                                    error="page_id and text are required",
                                    code="invalid_arguments")
        blocks = _blocks_from_text(text)
        # Notion caps a single append at 100 blocks per call. Chunk.
        appended = 0
        for i in range(0, len(blocks), 100):
            chunk = blocks[i:i + 100]
            status, body = _request(
                "PATCH", f"/blocks/{page_id}/children",
                self._token, body={"children": chunk},
            )
            if status != 200:
                return self._http_error("append failed", status, body)
            appended += len(chunk)
        return connector_result("ok", blocks_appended=appended)

    def _do_create_page(self, args: Dict[str, Any]) -> Dict[str, Any]:
        parent_id = str(args.get("parent_page_id") or "").strip()
        title = str(args.get("title") or "").strip()
        if not parent_id or not title:
            return connector_result(
                "error",
                error="parent_page_id and title are required",
                code="invalid_arguments",
            )
        body_text = args.get("body") or ""
        children = _blocks_from_text(body_text) if str(body_text).strip() else []
        # First 100 children go with the create call; the rest get appended.
        first_batch = children[:100]
        remaining = children[100:]
        payload = {
            "parent": {"type": "page_id", "page_id": parent_id},
            "properties": {
                "title": {
                    "title": [{"type": "text",
                               "text": {"content": title[:2000]}}]
                }
            },
        }
        if first_batch:
            payload["children"] = first_batch
        status, data = _request("POST", "/pages", self._token, body=payload)
        if status != 200:
            return self._http_error("page create failed", status, data)
        new_id = data.get("id", "")
        # Tail-append any remaining blocks past the 100-child create cap.
        appended_extra = 0
        if new_id and remaining:
            for i in range(0, len(remaining), 100):
                chunk = remaining[i:i + 100]
                s2, b2 = _request(
                    "PATCH", f"/blocks/{new_id}/children",
                    self._token, body={"children": chunk},
                )
                if s2 != 200:
                    return self._http_error("created page but tail-append "
                                            "failed", s2, b2,
                                            partial_id=new_id)
                appended_extra += len(chunk)
        return connector_result(
            "ok",
            page_id=new_id,
            url=data.get("url", ""),
            initial_blocks=len(first_batch),
            extra_blocks=appended_extra,
        )

    def _do_add_to_database(self, args: Dict[str, Any]) -> Dict[str, Any]:
        db_id = str(args.get("database_id") or "").strip()
        title = str(args.get("title") or "").strip()
        props_raw = args.get("properties") or {}
        if not db_id or not title:
            return connector_result(
                "error",
                error="database_id and title are required",
                code="invalid_arguments",
            )
        # Discover the database's property schema so we know which Notion
        # type to wrap each value in. Without this we'd have to guess
        # text vs select vs date and end up with 400s.
        status, db_meta = _request("GET", f"/databases/{db_id}", self._token)
        if status != 200:
            return self._http_error("database fetch failed", status, db_meta)
        schema = db_meta.get("properties") or {}
        # Find the title column — its name varies ("Name", "Task", "Title"…).
        title_key = next(
            (k for k, v in schema.items() if v.get("type") == "title"), None
        )
        if not title_key:
            return connector_result(
                "error",
                error="database has no title column — can't add a row",
                code="no_title_column",
            )
        props_payload: Dict[str, Any] = {
            title_key: {
                "title": [{"type": "text",
                           "text": {"content": title[:2000]}}]
            }
        }
        # Wrap the other named values per the database's actual type.
        for name, value in (props_raw or {}).items():
            spec = schema.get(name)
            if not spec:
                # Unknown property — skip rather than 400 the whole row.
                continue
            wrapped = _wrap_property_value(spec.get("type"), value)
            if wrapped is not None:
                props_payload[name] = wrapped
        payload = {
            "parent": {"type": "database_id", "database_id": db_id},
            "properties": props_payload,
        }
        s, data = _request("POST", "/pages", self._token, body=payload)
        if s != 200:
            return self._http_error("database row create failed", s, data)
        return connector_result(
            "ok",
            page_id=data.get("id", ""),
            url=data.get("url", ""),
            title_column=title_key,
            properties_set=list(props_payload.keys()),
        )

    # ---- error helper -------------------------------------------------------

    def _http_error(self, what: str, status: int, body: Dict[str, Any],
                    **extra: Any) -> Dict[str, Any]:
        """Common error result for any failed Notion call. Surfaces Notion's
        own message when present so the model can tell the user a real cause
        (e.g. 'object_not_found' usually means the page isn't shared)."""
        msg = (body.get("message") or body.get("error") or "").strip()
        code = (body.get("code") or "").strip()
        # Decode the most common shape so the model can react usefully.
        hint = ""
        if code == "object_not_found" or status == 404:
            hint = (" Most likely: that page/database isn't shared with the "
                    "Iris integration yet. Open it in Notion -> Share -> Add "
                    "connections -> Iris.")
        elif code == "unauthorized" or status == 401:
            hint = (" Token rejected — say 'set up notion' to refresh it.")
        return connector_result(
            "error",
            error=f"{what}: {msg or f'http {status}'}{hint}",
            http_status=status,
            notion_code=code,
            **extra,
        )


def _result_title(node: Dict[str, Any]) -> str:
    """Pull a human-readable title out of a Notion API result. Pages stash
    their title inside one of the properties; databases keep it at the top
    level. We try both shapes so search results render uniformly."""
    # Database: top-level `title` is a list of rich_text.
    title_runs = node.get("title")
    if isinstance(title_runs, list) and title_runs:
        text = "".join(str(t.get("plain_text") or "") for t in title_runs)
        if text.strip():
            return text.strip()
    # Page: walk properties for the one that's of type 'title'.
    props = node.get("properties") or {}
    for spec in props.values():
        if isinstance(spec, dict) and spec.get("type") == "title":
            runs = spec.get("title") or []
            text = "".join(str(t.get("plain_text") or "") for t in runs)
            if text.strip():
                return text.strip()
    return "(untitled)"


def _wrap_property_value(prop_type: str, value: Any) -> Optional[Dict[str, Any]]:
    """Wrap a simple Python value in Notion's expected property shape based
    on the database's declared property type. None means we don't know how
    to wrap this combination — caller skips it rather than send junk."""
    t = (prop_type or "").strip().lower()
    if value is None:
        return None
    if t == "rich_text":
        s = str(value)[:2000]
        return {"rich_text": [{"type": "text", "text": {"content": s}}]}
    if t == "number":
        try:
            return {"number": float(value)}
        except Exception:
            return None
    if t == "checkbox":
        return {"checkbox": bool(value)}
    if t == "url":
        return {"url": str(value)}
    if t == "email":
        return {"email": str(value)}
    if t == "phone_number":
        return {"phone_number": str(value)}
    if t == "select":
        # Single select — value is the option name.
        return {"select": {"name": str(value)}}
    if t == "multi_select":
        # Accept a list or comma-separated string.
        if isinstance(value, str):
            names = [v.strip() for v in value.split(",") if v.strip()]
        else:
            try:
                names = [str(v) for v in value]
            except Exception:
                return None
        return {"multi_select": [{"name": n} for n in names]}
    if t == "date":
        # Value should be an ISO date string (YYYY-MM-DD or full ISO 8601).
        return {"date": {"start": str(value)}}
    if t == "status":
        return {"status": {"name": str(value)}}
    if t == "people":
        # People expects user IDs; comma-list or actual list.
        if isinstance(value, str):
            ids = [v.strip() for v in value.split(",") if v.strip()]
        else:
            try:
                ids = [str(v) for v in value]
            except Exception:
                return None
        return {"people": [{"id": i} for i in ids]}
    if t == "files":
        # Expect a list of URLs (external file refs).
        if isinstance(value, str):
            urls = [value]
        else:
            try:
                urls = [str(v) for v in value]
            except Exception:
                return None
        return {
            "files": [
                {"type": "external", "name": u.rsplit("/", 1)[-1] or "file",
                 "external": {"url": u}}
                for u in urls
            ]
        }
    # Title type is set by the caller (not from the user-supplied dict).
    if t == "title":
        return None
    return None
