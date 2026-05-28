"""Deterministic intent classifier — maps obvious single-intent commands to a
Step (tool + args) with **no LLM call**, so common requests never burn a model
turn or hit a rate limit.

Strict, anchored regex patterns. Conservative on purpose: false positives are
worse than misses (a miss just falls through to the LLM path). When a pattern
doesn't clearly match, return None and let the planner / LLM take over.

Covers high-impact, high-confidence intents in Phase 1:
  - system volume (set/get/mute/toggle)
  - Discord voice (mute/deafen/toggle)
  - Microsoft To Do task add
  - Google Docs/Sheets/Slides create
  - Google Drive upload (with explicit path)
  - Email compose with explicit recipient + body

Author: Konstantin Markov
"""
from __future__ import annotations

import re
from typing import Optional

from .plan import Step


class Classifier:
    """Try to deterministically map a normalized command to a Step."""

    def classify(self, text: str) -> Optional[Step]:
        t = (text or "").strip()
        if not t:
            return None
        lower = t.lower()

        # ---- system volume ------------------------------------------------
        # "set volume to 30", "volume 30", "volume to 30%"
        m = re.search(r"\b(?:set\s+(?:the\s+)?volume(?:\s+to)?|volume(?:\s+to)?)\s+(\d{1,3})\s*%?\b", lower)
        if m and "discord" not in lower:
            pct = max(0, min(100, int(m.group(1))))
            return Step(tool="volume_set", args={"percent": pct}, layer="connector",
                        description=f"set system volume to {pct}%")
        # "what(’s) the volume", "current volume", "is muted"
        if re.search(r"\b(?:what(?:'s| is)|how (?:loud|much)|current)\b.*\bvolume\b", lower) and "discord" not in lower:
            return Step(tool="volume_get", args={}, layer="connector",
                        description="read system volume")
        # mute / unmute (system audio, not Discord)
        if "discord" not in lower and re.fullmatch(r"\s*(unmute|mute)(?:\s+(?:the\s+)?(?:system|audio|sound|volume|computer))?\s*", lower):
            return Step(tool="volume_mute",
                        args={"muted": "unmute" not in lower},
                        layer="connector",
                        description="mute/unmute system audio")
        # "toggle (the )?mute"
        if "discord" not in lower and re.fullmatch(r"\s*toggle\s+(?:the\s+)?mute\s*", lower):
            return Step(tool="volume_toggle_mute", args={}, layer="connector",
                        description="toggle system mute")

        # ---- Discord voice ------------------------------------------------
        if "discord" in lower:
            if re.search(r"\b(?:un)?mute\b.*\bdiscord\b|\bdiscord\b.*\b(?:un)?mute\b", lower) and "deafen" not in lower:
                if re.search(r"\btoggle\b", lower):
                    return Step(tool="discord_toggle_mute", args={}, layer="connector",
                                description="toggle Discord mute")
                return Step(tool="discord_mute",
                            args={"muted": "unmute" not in lower},
                            layer="connector",
                            description="set Discord mute")
            if re.search(r"\b(?:un)?deafen\b", lower):
                if re.search(r"\btoggle\b", lower):
                    return Step(tool="discord_toggle_deafen", args={}, layer="connector",
                                description="toggle Discord deafen")
                return Step(tool="discord_deafen",
                            args={"deafened": "undeafen" not in lower},
                            layer="connector",
                            description="set Discord deafen")

        # ---- Microsoft To Do ---------------------------------------------
        # "add a task: buy milk" / "add task buy milk" / "remind me to buy milk"
        m = (re.search(r"\badd\s+(?:a\s+)?task[:\s]+(.+)$", lower)
             or re.search(r"\bremind\s+me\s+to\s+(.+)$", lower)
             or re.search(r"\bnew\s+task[:\s]+(.+)$", lower))
        if m:
            title = m.group(1).strip().rstrip(".!?")
            return Step(tool="todo_add", args={"title": title}, layer="connector",
                        description=f"add task: {title}")

        # ---- Google Docs / Sheets / Slides create -------------------------
        m = re.search(
            r"\b(?:make|create|new)\s+(?:a\s+)?(?:new\s+)?(?:google\s+)?"
            r"(doc(?:ument)?|sheet|spreadsheet|slide(?:show)?|presentation|deck)\b"
            r"(?:\s+(?:titled|called|named)\s+(.+?))?(?:\s+with\s+|\s*$)",
            lower,
        )
        if m:
            kind = m.group(1)
            title = (m.group(2) or "Untitled").strip().rstrip(".!?").strip("'\"")
            if kind.startswith("doc"):
                return Step(tool="gdocs_create", args={"title": title}, layer="connector",
                            description=f"create Google Doc: {title}")
            if kind in ("sheet", "spreadsheet"):
                return Step(tool="sheets_create", args={"title": title}, layer="connector",
                            description=f"create Google Sheet: {title}")
            return Step(tool="slides_create", args={"title": title}, layer="connector",
                        description=f"create Google Slides: {title}")

        # ---- Google Drive upload (explicit local path) --------------------
        m = re.search(
            r"\bupload\s+(?P<path>[^\s\"]+|\"[^\"]+\"|'[^']+')\s+to\s+(?:my\s+)?(?:google\s+)?drive\b",
            t, flags=re.IGNORECASE,
        )
        if m:
            path = m.group("path").strip("'\"")
            return Step(tool="drive_upload", args={"path": path}, layer="connector",
                        description=f"upload to Drive: {path}")

        # ---- Email compose (recipient + body, simple phrasing) -----------
        # Two forms, both produce an outlook_compose Step that the
        # orchestrator MAY rewrite to gmail_send/ms_mail_send based on the
        # user's default_send_via preference (and resolve a name → email
        # from memory if recipient isn't an @ address):
        #
        #   "email <addr> saying <body>"  — explicit address
        #   "email <Name> saying <body>"  — a named recipient (the
        #     orchestrator looks up Name in memory)
        # Excluded: pronouns/articles ("me", "him", "the boss" — too
        # ambiguous; let those fall through to higher tiers).
        m = re.search(
            r"\bemail\s+(?P<to>\S+@\S+\.\S+|[A-Za-z][A-Za-z0-9._\-]*)"
            r"\s+(?:saying|with message|that says)\s+(?P<body>.+)$",
            t, flags=re.IGNORECASE,
        )
        if m:
            recipient = m.group("to").strip(",.;")
            # Reject ambiguous "recipients" so we don't grab the wrong word.
            _STOP_NAMES = {"me", "him", "her", "them", "us", "the", "a", "an"}
            if "@" in recipient or recipient.lower() not in _STOP_NAMES:
                return Step(
                    tool="outlook_compose",
                    args={"recipient": recipient,
                          "body": m.group("body").strip()},
                    layer="connector",
                    description="compose email draft",
                )

        # ---- contact lookup ("what's Dani's email", "find Dani's email") --
        # Read intent, NOT a compose. The orchestrator answers from memory.
        m = re.search(
            r"\b(?:find|look\s*up|what(?:'s| is)|tell\s+me|give\s+me|"
            r"do\s+you\s+know)\b[^?]*?\b(?P<name>[A-Za-z][A-Za-z0-9._\-]*)"
            r"(?:'s|s|)\s+(?:email|email\s+address|address)\b",
            t, flags=re.IGNORECASE,
        )
        if m:
            name = m.group("name")
            # Reject ambiguous "names".
            if name.lower() not in {"my", "me", "your", "the", "a", "an", "his",
                                    "her", "their", "our"}:
                return Step(
                    tool="iris_lookup_contact",
                    args={"name": name},
                    layer="touchless",
                    description=f"look up {name}'s email in memory",
                )

        # ---- sticky user preferences (saved to memory, 0 tokens) ----------
        # "always send from gmail", "use outlook by default", "set my default
        # sender to gmail" — saved as preference(default_send_via=...). The
        # planner's recall layer injects it on every future Phase-2 prompt.
        m = re.search(
            r"\b(?:always|only|by\s+default|set\s+(?:my\s+)?default)\b.*?\b"
            r"(?:send|email)\b.*?\b(?:from|via|using|with)\s+"
            r"(?:my\s+)?(?P<sender>gmail|google|outlook|microsoft|ms|m365|hotmail|exchange)\b",
            lower,
        )
        if m is None:
            m = re.search(
                r"\bset\s+(?:my\s+)?default\s+sender\s+to\s+"
                r"(?P<sender>gmail|google|outlook|microsoft|ms|m365|hotmail|exchange)\b",
                lower,
            )
        if m is None:
            # "use outlook by default" / "always use gmail" — sender named
            # without an explicit 'send' verb. We assume "use" + sender =
            # use as default email sender (the most common interpretation).
            m = re.search(
                r"\b(?:always\s+use|use)\s+(?:my\s+)?"
                r"(?P<sender>gmail|google|outlook|microsoft|ms|m365|hotmail|exchange)"
                r"\b(?:\s+by\s+default)?",
                lower,
            )
        if m is not None:
            tool_for = {
                "gmail": "gmail_send", "google": "gmail_send",
                "outlook": "ms_mail_send", "microsoft": "ms_mail_send",
                "ms": "ms_mail_send", "m365": "ms_mail_send",
                "hotmail": "ms_mail_send", "exchange": "ms_mail_send",
            }
            sender = tool_for.get(m.group("sender"))
            if sender:
                return Step(
                    tool="iris_set_preference",
                    args={"kind": "preference", "key": "default_send_via",
                          "value": sender},
                    layer="touchless",
                    description=f"remember to send via {sender} by default",
                )

        # "always search contacts in my .edu" / "only look up contacts in
        # gmail / outlook / school". Saved as default_contact_account.
        m = re.search(
            r"\b(?:always|only|by\s+default)\b.*?\b"
            r"(?:search|look\s*up|find)\b.*?\bcontacts?\b.*?\b(?:in|from|via|using)\s+"
            r"(?:my\s+)?(?P<acct>gmail|google|outlook|microsoft|ms|m365|hotmail|exchange|"
            r"school|work|personal|\.?edu|university)\b",
            lower,
        )
        if m is not None:
            return Step(
                tool="iris_set_preference",
                args={"kind": "preference", "key": "default_contact_account",
                      "value": m.group("acct")},
                layer="touchless",
                description=f"remember to search contacts in {m.group('acct')}",
            )

        return None
