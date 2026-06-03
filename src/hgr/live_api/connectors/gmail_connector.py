"""Gmail connector — API-first email read/send via the Gmail API.

Deep, specific tasks ("email X saying Y", "what are my unread emails",
"find the message from the bank") done with one API call. Dormant until
the shared GoogleClient is authorized (see google_client.py).

Author: Konstantin Markov
"""
from __future__ import annotations

import base64
import re
from email.mime.text import MIMEText
from typing import Any, Dict, List, Optional

from .base import Connector, connector_result
from .google_client import GoogleClient


_HTML_TAG_RE = re.compile(r"<[^>]+>")
_HTML_ENTITY_RE = re.compile(r"&(?:nbsp|amp|lt|gt|quot|apos|#\d+);")
_WS_COLLAPSE_RE = re.compile(r"\s+")


def _strip_html(html: str) -> str:
    text = _HTML_TAG_RE.sub(" ", html or "")
    text = _HTML_ENTITY_RE.sub(lambda m: {
        "&nbsp;": " ", "&amp;": "&", "&lt;": "<", "&gt;": ">",
        "&quot;": '"', "&apos;": "'",
    }.get(m.group(0), " "), text)
    return _WS_COLLAPSE_RE.sub(" ", text).strip()


def _header(headers: List[Dict[str, str]], name: str) -> str:
    """Pull a single header value (case-insensitive) from a Gmail payload."""
    name_l = name.lower()
    for h in headers or []:
        if (h.get("name") or "").lower() == name_l:
            return h.get("value") or ""
    return ""


def _split_from(raw: str) -> tuple:
    """Parse a Gmail 'From' header. Returns (display_name, email_address).
    Handles 'Dani M <dani@x>', '<dani@x>', 'dani@x'. Quotes/escapes
    intentionally not exhaustively parsed — name is informational."""
    raw = (raw or "").strip()
    if not raw:
        return ("", "")
    if "<" in raw and ">" in raw:
        name = raw.split("<", 1)[0].strip().strip('"').strip()
        addr = raw.split("<", 1)[1].split(">", 1)[0].strip()
        return (name, addr)
    if "@" in raw:
        return ("", raw)
    return (raw, "")


def _walk_for_body(payload: Dict[str, Any]) -> str:
    """Gmail nests bodies in mimeType=text/plain (preferred) or text/html
    leaves, sometimes deeply for multipart messages. Walk the tree and
    return whichever readable text we find first (plain wins over html)."""
    if not payload:
        return ""
    mime = (payload.get("mimeType") or "").lower()
    body = payload.get("body") or {}
    data = body.get("data") or ""

    def _decode(b64: str) -> str:
        try:
            return base64.urlsafe_b64decode(b64 + "=" * (-len(b64) % 4)).decode(
                "utf-8", errors="ignore")
        except Exception:
            return ""

    if data and mime == "text/plain":
        return _decode(data)

    plain = ""
    html = ""
    for part in (payload.get("parts") or []):
        found = _walk_for_body(part)
        pmime = (part.get("mimeType") or "").lower()
        if pmime == "text/plain" and found and not plain:
            plain = found
        elif pmime == "text/html" and found and not html:
            html = found
        else:
            # Nested multipart — could contain either.
            if found and not plain:
                plain = found if "text/plain" in pmime else plain
            if found and not html:
                html = found if "text/html" in pmime else html
    if plain:
        return plain
    if html:
        return _strip_html(html)
    if data and mime == "text/html":
        return _strip_html(_decode(data))
    return ""


def _format_email_summary(msgs: List[Dict[str, Any]], *,
                          unread_only: bool,
                          max_arg: int) -> str:
    """Render a faithful, casual summary of an email list.

    Every sender, subject, and snippet comes verbatim from the message
    array. No LLM rephrasing — the model has hallucinated demo emails
    here, so the deterministic format is what the user actually sees."""
    count = len(msgs or [])
    kind = "unread email" if unread_only else "email"
    if count == 0:
        if unread_only:
            return "You're all caught up — no unread emails."
        return "No emails in your inbox."
    if count == 1:
        head = f"You've got one {kind}:"
    else:
        head = f"You've got {count} {kind}s — here they are:"
        if max_arg and count >= max_arg:
            head = (f"Showing the {count} most recent {kind}s "
                    f"(cap is {max_arg} — say 'show more' to dig deeper):")
    lines: List[str] = [head]
    for i, m in enumerate(msgs, start=1):
        sender = (str(m.get("from_name") or "").strip()
                  or str(m.get("from") or "").strip()
                  or "unknown sender")
        subject = str(m.get("subject") or "").strip() or "(no subject)"
        snippet = str(m.get("snippet") or m.get("preview")
                      or m.get("body_text") or "").strip()
        if snippet:
            snippet = " ".join(snippet.replace("\r", " ")
                               .replace("\n", " ").split())
            if len(snippet) > 140:
                snippet = snippet[:137].rstrip() + "..."
            lines.append(f"{i}. {sender} — \"{subject}\". {snippet}")
        else:
            lines.append(f"{i}. {sender} — \"{subject}\".")
    lines.append("Want me to open any of them or dig deeper?")
    return "\n".join(lines)


class GmailConnector(Connector):
    id = "gmail"

    def __init__(self, client: GoogleClient | None = None) -> None:
        self._client = client or GoogleClient.shared()

    def _svc(self):
        return self._client.service("gmail", "v1")

    def available(self) -> bool:
        # "Available at all" — we have SOME valid Google grant. Per-tool
        # scope checks live in execute() so the user gets a precise
        # error ("Reading email needs gmail.readonly") instead of a
        # generic "Gmail not authorized" when their grant is partial.
        try:
            return self._client.ready()
        except Exception:
            return False

    def _has_readonly(self) -> bool:
        try:
            return self._client.has_scope(
                "https://www.googleapis.com/auth/gmail.readonly")
        except Exception:
            return False

    def _has_send(self) -> bool:
        try:
            return self._client.has_scope(
                "https://www.googleapis.com/auth/gmail.send")
        except Exception:
            return False

    def tools(self) -> List[Dict[str, Any]]:
        def fn(name, desc, props=None, required=None):
            return {"type": "function", "name": name, "description": desc,
                    "parameters": {"type": "object", "properties": props or {},
                                   "required": required or [],
                                   "additionalProperties": False}}
        return [
            fn("gmail_send",
               "Send an email via Gmail. Sends immediately — confirm intent first. "
               "Always include a concise subject.",
               {"to": {"type": "string", "description": "Recipient email address."},
                "subject": {"type": "string"},
                "body": {"type": "string"}},
               ["to", "subject", "body"]),
            fn("gmail_list",
               "List recent Gmail inbox messages. Set unread_only=true for "
               "INBOX + UNREAD only. Returns {messages: [{id, from, "
               "from_name, subject, received, snippet}], count}. `count` is "
               "the EXACT number of messages returned — quote it directly, "
               "never round or guess. Set include_body=true to ALSO fetch "
               "each full body (HTML stripped, capped at 2 KB) inline as "
               "body_text — required for any real 'summarize/read my "
               "emails' request so the reply has substance. Set max=50 "
               "(the cap) for any 'summarize ALL my unread' request; the "
               "default of 10 silently truncates a real inbox. NEVER "
               "invent message content not in the returned array — if "
               "count=0, say there are no unread emails. Works for the "
               "user's actual Gmail inbox; prefer this over ms_mail_list "
               "when the connected MS account is a personal MSA (Graph "
               "contacts/inbox are incomplete on those).",
               {"max": {"type": "integer",
                        "description": "Max messages 1-50 (default 10). "
                                       "Use 50 for 'summarize unread'."},
                "unread_only": {"type": "boolean", "default": False},
                "query": {"type": "string",
                          "description": "Optional Gmail-style search query "
                                         "(e.g. 'from:dani@x', "
                                         "'newer_than:1d')."},
                "include_body": {"type": "boolean", "default": False}}),
            fn("gmail_read",
               "Fetch one Gmail message by id (from gmail_list). Returns the "
               "full headers + body_text (HTML stripped).",
               {"id": {"type": "string"}}, ["id"]),
        ]

    def execute(self, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        # Per-tool scope precondition: read tools need gmail.readonly,
        # send tools need gmail.send. If the user granted a partial
        # set of scopes (very common — Google's consent screen lets
        # them uncheck individual boxes), we want a precise, actionable
        # error like "you're connected to Google but didn't grant
        # 'Read your email' — reconnect and tick that box" instead of
        # a generic "Gmail not authorized" that makes the user think
        # nothing is connected.
        need_readonly = name in ("gmail_list", "gmail_read")
        need_send = name == "gmail_send"
        if need_readonly and not self._has_readonly():
            return connector_result(
                "error",
                error=("Gmail is connected, but you didn't grant the "
                       "'Read your email' (gmail.readonly) scope on "
                       "the consent screen. Click 'Connect Gmail' "
                       "again and make sure every checkbox — "
                       "especially 'Read your email' — is ticked."),
                code="scope_missing",
                missing_scope="gmail.readonly")
        if need_send and not self._has_send():
            return connector_result(
                "error",
                error=("Gmail is connected, but you didn't grant the "
                       "'Send email' (gmail.send) scope. Reconnect "
                       "with that box ticked."),
                code="scope_missing",
                missing_scope="gmail.send")
        svc = self._svc()
        if svc is None:
            return connector_result(
                "error",
                error=("Gmail isn't connected. Click 'Connect Gmail' "
                       "to grant access."),
                code="not_connected")
        try:
            if name == "gmail_send":
                to = str(args.get("to") or "").strip()
                subject = str(args.get("subject") or "")
                body = str(args.get("body") or "")
                if not to:
                    return connector_result("error", error="'to' is required")
                msg = MIMEText(body)
                msg["to"] = to
                msg["subject"] = subject
                raw = base64.urlsafe_b64encode(msg.as_bytes()).decode()
                sent = svc.users().messages().send(userId="me", body={"raw": raw}).execute()
                return connector_result("ok", sent=True, id=sent.get("id"), to=to)

            if name == "gmail_list":
                max_n = max(1, min(50, int(args.get("max") or 10)))
                include_body = bool(args.get("include_body"))
                # Build the Gmail-style query. Unread+INBOX is the morning-
                # briefing default; the LLM can override with a `query` arg.
                user_q = str(args.get("query") or "").strip()
                if user_q:
                    q = user_q
                elif args.get("unread_only"):
                    q = "in:inbox is:unread"
                else:
                    q = "in:inbox"
                listed = svc.users().messages().list(
                    userId="me", q=q, maxResults=max_n).execute()
                ids = [m.get("id") for m in (listed.get("messages") or [])
                       if m.get("id")]
                # `format=metadata` is enough for headers + snippet; we only
                # promote to `full` when include_body is on.
                fmt = "full" if include_body else "metadata"
                hdrs = ["From", "Subject", "Date"]
                out_msgs: List[Dict[str, Any]] = []
                for mid in ids:
                    msg = svc.users().messages().get(
                        userId="me", id=mid, format=fmt,
                        metadataHeaders=hdrs).execute()
                    headers = (msg.get("payload") or {}).get("headers") or []
                    from_raw = _header(headers, "From")
                    from_name, from_addr = _split_from(from_raw)
                    out: Dict[str, Any] = {
                        "id": msg.get("id"),
                        "from": from_addr,
                        "from_name": from_name,
                        "subject": _header(headers, "Subject"),
                        "received": _header(headers, "Date"),
                        "snippet": msg.get("snippet"),
                    }
                    if include_body:
                        body_text = _walk_for_body(msg.get("payload") or {})
                        out["body_text"] = body_text[:2000]
                    out_msgs.append(out)
                # Deterministic, faithful summary built directly from the
                # real message array. The LLM has been caught fabricating
                # demo emails when given freedom to "summarize"; emitting
                # this summary verbatim guarantees the user sees real
                # senders and subjects.
                summary = _format_email_summary(
                    out_msgs,
                    unread_only=bool(args.get("unread_only")),
                    max_arg=max_n,
                )
                return connector_result("ok", count=len(out_msgs),
                                        messages=out_msgs,
                                        summary=summary)

            if name == "gmail_read":
                mid = str(args.get("id") or "").strip()
                if not mid:
                    return connector_result("error", error="'id' is required")
                msg = svc.users().messages().get(
                    userId="me", id=mid, format="full").execute()
                headers = (msg.get("payload") or {}).get("headers") or []
                from_raw = _header(headers, "From")
                from_name, from_addr = _split_from(from_raw)
                return connector_result(
                    "ok",
                    id=msg.get("id"),
                    from_=from_addr,
                    from_name=from_name,
                    to=_header(headers, "To"),
                    subject=_header(headers, "Subject"),
                    received=_header(headers, "Date"),
                    snippet=msg.get("snippet"),
                    body_text=_walk_for_body(msg.get("payload") or {})[:5000],
                )
        except Exception as exc:
            return connector_result("error", error=f"{type(exc).__name__}: {exc}")
        return connector_result("error", error=f"unknown gmail tool: {name}", code="no_handler")
