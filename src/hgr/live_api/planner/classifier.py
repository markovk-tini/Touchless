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
        # Title capture stops at common chain words ("and write/include/add/
        # fill/put/...", "then", commas) so requests like 'create doc called
        # X AND write Y' don't grab the whole tail as the title — those
        # chain forms should yield to multi-action / Tier 2.
        _TITLE_STOP = (r"\s+(?:and|then|,|;|"
                       r"with\s+(?:the\s+)?(?:weather|email|content|text|body|"
                       r"morning|debrief|summary|briefing|brief))\b")
        m = re.search(
            r"\b(?:make|create|new)\s+(?:a\s+)?(?:new\s+)?(?:google\s+)?"
            r"(doc(?:ument)?|sheet|spreadsheet|slide(?:show)?|presentation|deck)\b"
            r"(?:\s+(?:titled|called|named)\s+(.+?))?"
            rf"(?:{_TITLE_STOP}|\s*$)",
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

        # ---- weather ("what's the weather", "is it raining", "weather in X")
        # Free, no-key wttr.in lookup. Auto-detects location when none given.
        # "weather in <place>" / "weather for <place>" / "<place> weather"
        m = re.search(
            r"\b(?:weather|forecast|temperature|temp)\s+"
            r"(?:in|for|at|of|near|around)\s+(?P<loc>[A-Za-z][A-Za-z0-9 ,.\-'’]{1,80})\??$",
            t, flags=re.IGNORECASE,
        )
        if m:
            return Step(
                tool="weather_get",
                args={"location": m.group("loc").strip(" ?.,")},
                layer="touchless",
                description=f"weather in {m.group('loc').strip()}",
            )
        # "what's/how's the weather", "is it raining/snowing/hot/cold",
        # "what's the temperature", "should I bring an umbrella"
        m = re.search(
            r"\b("
            r"(?:what(?:[’'´]?s|s'?s| is)|how(?:[’'´]?s| is)|tell\s+me|"
            r"give\s+me)\s+(?:about\s+)?(?:the\s+)?"
            r"(?:weather|forecast|temperature|temp|conditions?)|"
            r"is\s+it\s+(?:raining|snowing|cold|hot|warm|chilly|nice\s+out|"
            r"freezing|sunny|cloudy|going\s+to\s+rain|going\s+to\s+snow)|"
            r"(?:do\s+i\s+need|should\s+i\s+(?:bring|wear|grab))\s+"
            r"(?:an?\s+)?(?:umbrella|jacket|coat|sweater)"
            r")\b",
            t, flags=re.IGNORECASE,
        )
        if m:
            return Step(
                tool="weather_get",
                args={},
                layer="touchless",
                description="local weather",
            )

        # ---- contact forget ("forget Dani", "forget Dani's email",
        # "delete Vesko from memory", "you can forget about Mariya") ----
        # Lets the user purge memory facts without the CLI.
        m = re.search(
            r"\b(?:forget|delete|remove|wipe)\b"
            r"(?:\s+about|\s+everything\s+about)?\s+"
            r"(?P<name>[A-Za-z][A-Za-z0-9._\-]*)"
            r"(?:'s|s)?\s*(?:email|address|info|details|number|"
            r"from\s+(?:memory|your\s+memory)|address\s+book)?",
            t, flags=re.IGNORECASE,
        )
        if m:
            name = m.group("name")
            _STOP = {"my", "me", "your", "the", "a", "an", "his", "her",
                     "their", "our", "all", "everything", "it", "that"}
            if name.lower() not in _STOP:
                return Step(
                    tool="iris_forget_contact",
                    args={"name": name},
                    layer="touchless",
                    description=f"forget {name} in memory",
                )

        # ---- contact remember ("X is for Vesko and Mariya", "Vesko at X") --
        # Lets the user TEACH Iris facts via Tier 1, no realtime/LLM needed.
        # Supports one or many names per email so 'shared address' patterns
        # like 'X is for Vesko and Mariya' record both bindings.
        _EMAIL = r"(?P<email>[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,})"
        _NAMES = r"(?P<names>[A-Za-z][a-zA-Z]+(?:\s*(?:,|and)\s*[A-Za-z][a-zA-Z]+)*)"
        remember_patterns = (
            # "<email> is for [both] <name(s)>" / "<email> is the email of <names>"
            rf"{_EMAIL}\s+is\s+(?:for|the\s+email\s+(?:for|of))\s+(?:both\s+)?{_NAMES}\b",
            # "<email> is <name(s)>'s email"
            rf"{_EMAIL}\s+is\s+{_NAMES}(?:'s|s)?\s+email\b",
            # "remember <name(s)> (is|at|=) <email>" / "<name(s)>'s email is <email>"
            rf"(?:remember\s+)?{_NAMES}(?:'s|s)?\s+(?:email\s+)?(?:is|at|=)\s+{_EMAIL}",
            # "<name(s)> share/use <email>"
            rf"{_NAMES}\s+(?:share|use|have\s+the\s+email)\s+{_EMAIL}",
        )
        for pat in remember_patterns:
            m = re.search(pat, t, flags=re.IGNORECASE)
            if not m:
                continue
            raw_names = m.group("names")
            names = re.findall(r"[A-Za-z][a-zA-Z]+", raw_names)
            # Skip stopwords masquerading as names.
            stop = {"and", "is", "for", "the", "of", "both", "at", "or",
                    "with", "to", "from", "by"}
            names = [n for n in names if n.lower() not in stop]
            if names:
                return Step(
                    tool="iris_remember_contact",
                    args={"names": names, "email": m.group("email")},
                    layer="touchless",
                    description=f"remember {', '.join(names)} = {m.group('email')}",
                )

        # ---- contact lookup ("what's Dani's email", "find Dani's email") --
        # Read intent, NOT a compose. The orchestrator answers from memory.
        # Verb form patterns are intentionally loose — common typos
        # ("whats's"), missing apostrophes ("whats"), smart quotes (’),
        # and conversational openers ("show me", "i need", "do you have")
        # all map to the same intent. False positives just answer 'I don't
        # have that in memory' which is harmless.
        m = re.search(
            r"\b(?:"
            r"find|look\s*up|"
            r"what(?:[’'´]?s|s[’'´]?s?|\s+is)|"
            r"where(?:[’'´]?s|\s+is)|"
            r"tell\s+me|give\s+me|show\s+me|"
            r"i\s+need|i\s+want|"
            r"do\s+you\s+(?:know|have)|"
            r"have\s+you\s+got|got\s+(?:the|a)"
            r")\b[^?]*?\b(?P<name>[A-Za-z][A-Za-z0-9._\-]*)"
            r"(?:[’'´]?s|s)?\s+(?:email|email\s+address|address)\b",
            t, flags=re.IGNORECASE,
        )
        if m:
            name = m.group("name")
            # Reject ambiguous "names".
            if name.lower() not in {"my", "me", "your", "the", "a", "an", "his",
                                    "her", "their", "our"}:
                # Only a PURE lookup. If the request also wants to send/
                # email/compose something, this is a chain — fall through
                # to Tier 2 so the full intent is planned together.
                tail = t[m.end():].lower()
                _SEND_WORDS = (" send ", " email ", " message ", " write ",
                               " draft ", " compose ", " reply ", " text ")
                if not any(w in tail for w in _SEND_WORDS):
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
