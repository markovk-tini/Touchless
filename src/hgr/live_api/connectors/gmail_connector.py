"""Gmail connector — API-first email read/send via the Gmail API.

Deep, specific tasks ("email X saying Y", "what are my unread emails",
"find the message from the bank") done with one API call. Dormant until
the shared GoogleClient is authorized (see google_client.py).

Author: Konstantin Markov
"""
from __future__ import annotations

import base64
from email.mime.text import MIMEText
from typing import Any, Dict, List

from .base import Connector, connector_result
from .google_client import GoogleClient


class GmailConnector(Connector):
    id = "gmail"

    def __init__(self, client: GoogleClient | None = None) -> None:
        self._client = client or GoogleClient.shared()

    def _svc(self):
        return self._client.service("gmail", "v1")

    def available(self) -> bool:
        try:
            return self._client.ready()
        except Exception:
            return False

    def tools(self) -> List[Dict[str, Any]]:
        def fn(name, desc, props=None, required=None):
            return {"type": "function", "name": name, "description": desc,
                    "parameters": {"type": "object", "properties": props or {},
                                   "required": required or [],
                                   "additionalProperties": False}}
        # Send-only: the connector requests only the gmail.send scope (free,
        # no paid verification), so reading/searching mail is intentionally
        # not offered — that would need the restricted gmail.readonly scope.
        return [
            fn("gmail_send",
               "Send an email via Gmail. Sends immediately — confirm intent first. "
               "Always include a concise subject.",
               {"to": {"type": "string", "description": "Recipient email address."},
                "subject": {"type": "string"},
                "body": {"type": "string"}},
               ["to", "subject", "body"]),
        ]

    def execute(self, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        svc = self._svc()
        if svc is None:
            return connector_result("error", error="Gmail not authorized", code="not_ready")
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
        except Exception as exc:
            return connector_result("error", error=f"{type(exc).__name__}: {exc}")
        return connector_result("error", error=f"unknown gmail tool: {name}", code="no_handler")
