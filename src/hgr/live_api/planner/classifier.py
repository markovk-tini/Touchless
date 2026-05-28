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
        # "email <addr> saying <body>" — strict to avoid mis-firing on read intents.
        m = re.search(
            r"\bemail\s+(?P<to>\S+@\S+\.\S+)\s+(?:saying|with message|that says)\s+(?P<body>.+)$",
            t, flags=re.IGNORECASE,
        )
        if m:
            return Step(
                tool="outlook_compose",
                args={"recipient": m.group("to").strip(",.;"),
                      "body": m.group("body").strip()},
                layer="connector",
                description="compose email draft",
            )

        return None
