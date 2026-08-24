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
from typing import Any, Dict, List, Optional

from .plan import Step


_OPEN_SUFFIX_RE = re.compile(
    r"\s*[,.]?\s+(?:and\s+(?:then\s+)?|then\s+|,\s*and\s+)"
    r"open(?:\s+it|\s+them|\s+the\s+(?:doc(?:ument)?|sheet|spreadsheet|"
    r"slide(?:show)?|presentation|deck|file|page))?"
    r"\s*[.?!]*$",
    re.IGNORECASE,
)


# Guard used by contact-forget / contact-search blocks to bail out when the
# utterance is really about a spreadsheet / doc / slide cell (e.g. "remove
# Email from cell A2", "what is the value of cell A2 in the sheet called
# testing"). Without this the (forget|delete|remove) NAME and 'who is X'
# / 'find X in contacts' regexes happily grab a bare column header or noun
# as a contact name.
#
# NOTE: 'cell' is intentionally guarded ONLY when it isn't 'cell phone /
# number / service / tower / reception' — those phrasings are legitimate
# contact-lookup info-types and must still reach the phone-lookup path.
_SHEET_CELL_CTX_RE = re.compile(
    r"\b[A-Z]{1,3}[0-9]{1,4}\b"                        # A1-style ref
    r"|\bvalue\s+of\s+cell\b"                          # 'value of cell'
    r"|\b(?:google\s+)?spreadsheet\b"
    r"|\b(?:google\s+)?sheets?\b"                      # sheet / sheets / google sheet
    r"|\bcell\b(?!\s+(?:phone|number|service|tower|reception))"
    r"|\brow\b|\bcolumn\b|\brange\b",
    re.IGNORECASE,
)


# Guard: names that end in `-task`, `-note`, `-doc`, `-sheet`, `-slide`,
# `-deck`, `-event`, `-meeting`, ... are almost never real people. They
# are artifact identifiers the user coined (e.g. 'drill-task', 'q4_note',
# 'onboarding-doc') and should fall through to the LLM planner which can
# resolve them against the session artifact tracker instead of being
# blindly forwarded to iris_forget_contact.
_ARTIFACT_NOUN_SUFFIXES = (
    "task", "tasks", "note", "notes",
    "doc", "docs", "document", "documents",
    "sheet", "sheets", "spreadsheet", "spreadsheets",
    "slide", "slides", "slideshow", "deck", "decks",
    "presentation", "presentations",
    "event", "events", "meeting", "meetings",
    "reminder", "reminders", "item", "items",
    "calendar", "form", "poll", "survey", "quiz",
    "page", "pages", "file", "files",
)
_ARTIFACT_SUFFIX_RE = re.compile(
    r"[-_](?:" + "|".join(_ARTIFACT_NOUN_SUFFIXES) + r")\b",
    re.IGNORECASE,
)

# Phone / SMS context words. If any of these appear in the utterance,
# the email_summary classifier must NOT claim the phrase — its 'messages?'
# alternation otherwise hijacks phone-context requests ("any messages on
# my phone", "read my texts") that belong to phone_link_read_recent.
_PHONE_CONTEXT_RE = re.compile(
    r"\b(?:on\s+(?:my|the|this|that)\s+phone|phone\s+messages?"
    r"|texts?|sms|imessages?)\b",
    re.IGNORECASE,
)


_CHAIN_OPEN_KIND_BY_TOOL = {
    "gdocs_create": "doc",
    "sheets_create": "sheet",
    "slides_create": "slideshow",
}


def strip_open_suffix(text: str) -> tuple[str, bool]:
    """If text ends with an 'and open it'/'then open the doc' suffix, strip
    it and return (clean_text, True). Otherwise return (text, False).

    Used by classify_chain() to peel a trailing open instruction before the
    body-extracting regex runs, so 'make a doc titled X with body Y and
    open it' yields body='Y' (not 'Y and open it')."""
    t = text or ""
    m = _OPEN_SUFFIX_RE.search(t)
    if not m:
        return t, False
    return t[: m.start()].rstrip(" ,.;:!?"), True


# ---- shared helpers --------------------------------------------------------

_NUMBER_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "couple": 2, "few": 3, "several": 4,
}


def _extract_email_account(text: str) -> Optional[str]:
    """Pull an account-name hint out of common phrasings.

    Catches: 'my gmail', 'in my work account', 'from my personal',
    'check work email', 'summarize my school inbox', 'unread on my
    yahoo'. Returns the matched word lowercased (substring-matched
    against Outlook store names by the connector). Returns None if no
    explicit account is named (caller reads across all accounts)."""
    t = (text or "").lower()
    # Known provider / role keywords. Order: more specific first so
    # 'work' wins over a generic 'mail'.
    candidates = [
        "gmail", "outlook", "exchange", "yahoo", "icloud", "hotmail",
        "live", "aol", "proton", "fastmail",
        "work", "school", "personal", "main", "primary",
    ]
    for word in candidates:
        # Match when the keyword is anchored as a possessive ("my X"),
        # in/on/from ("in my X", "on my X", "from my X"), or a direct
        # noun-phrase like "X email"/"X inbox"/"X account"/"X mail".
        pattern = (
            rf"\b(?:my|in\s+my|on\s+my|from\s+my|the)\s+{word}\b"
            rf"|"
            rf"\b{word}\s+(?:email|emails|inbox|mail|messages?|account)\b"
        )
        if re.search(pattern, t):
            return word
    return None


def _extract_forecast_days(text: str) -> Optional[int]:
    """Parse the forecast horizon from a phrase like 'forecast for the
    next four days', '5-day forecast', 'this week', 'this weekend'. Returns
    None when no horizon is implied (caller uses weather_get's default)."""
    t = (text or "").lower()
    # "this week" / "next week" → 7 days
    if re.search(r"\b(?:this|next|coming|upcoming|the)\s+week\b", t):
        return 7
    # "weekend" → 2 days
    if re.search(r"\b(?:this|next|the|over\s+the|for\s+the|coming|upcoming)?\s*weekend\b", t):
        return 2
    # "tomorrow" → 2 (today + tomorrow)
    if re.search(r"\btomorrow\b", t) and not re.search(r"\b(?:after|day\s+after)\s+tomorrow\b", t):
        return 2
    # "day after tomorrow" → 3
    if re.search(r"\b(?:after|day\s+after)\s+tomorrow\b", t):
        return 3
    # "next N days" / "N day forecast" / "N-day"
    m = re.search(
        r"\b(?:next\s+|coming\s+|upcoming\s+|for\s+|over\s+(?:the\s+)?(?:next\s+)?)?"
        r"(\d+|one|two|three|four|five|six|seven|eight|nine|ten|"
        r"couple(?:\s+of)?|few|several)"
        r"[-\s]+(?:day|night)s?\b",
        t,
    )
    if m:
        token = m.group(1)
        if token.isdigit():
            n = int(token)
        else:
            n = _NUMBER_WORDS.get(token.split()[0], 0)
        if 1 <= n <= 10:
            return n
    return None


def _parse_sheet_body(body: str) -> List[List[str]]:
    """Parse a body string (peeled by the sheets `with header/columns/rows/
    data/values` body-prefix) into a list of rows for sheets_create.

    Recognized conventions (case-insensitive):
      - 'A, B, C'                  -> [['A','B','C']]
      - 'A,B,C and D,E,F'          -> [['A','B','C'], ['D','E','F']]
      - 'row 1 A, B, C row 2 D, E' -> [['A','B','C'], ['D','E']]
      - 'A in A1 and B in B1'      -> [['A','B']] (positioned by column letter)
    """
    src = (body or "").strip()
    if not src:
        return []

    cell_addr_re = re.compile(
        r"\s*(?P<val>.+?)\s+in\s+(?P<col>[A-Z]+)(?P<row>\d+)\s*",
        re.IGNORECASE,
    )
    cell_segments = re.split(r"\s+and\s+|\s*,\s*", src)
    cell_matches = [cell_addr_re.fullmatch(seg) for seg in cell_segments if seg]
    if cell_segments and all(m is not None for m in cell_matches):
        by_row: Dict[int, Dict[int, str]] = {}
        for m in cell_matches:
            val = m.group("val").strip().strip("'\"")
            col_letters = m.group("col").upper()
            col_idx = 0
            for ch in col_letters:
                col_idx = col_idx * 26 + (ord(ch) - ord("A") + 1)
            row_idx = int(m.group("row"))
            by_row.setdefault(row_idx, {})[col_idx] = val
        rows_out: List[List[str]] = []
        for r_idx in sorted(by_row):
            cols = by_row[r_idx]
            max_col = max(cols)
            rows_out.append([cols.get(c, "") for c in range(1, max_col + 1)])
        if rows_out:
            return rows_out

    row_n_re = re.compile(r"\brow\s+\d+\b\s*", re.IGNORECASE)
    if row_n_re.search(src):
        parts = [p.strip() for p in row_n_re.split(src) if p.strip()]
        rows = [[c.strip().strip("'\"") for c in re.split(r",\s*|\s+", p)
                 if c.strip()] for p in parts]
        rows = [r for r in rows if r]
        if rows:
            return rows

    row_chunks = re.split(r"\s+and\s+", src)
    rows = [[c.strip().strip("'\"") for c in chunk.split(",") if c.strip()]
            for chunk in row_chunks if chunk.strip()]
    rows = [r for r in rows if r]
    return rows


class Classifier:
    """Try to deterministically map a normalized command to a Step."""

    def classify(self, text: str) -> Optional[Step]:
        t = (text or "").strip()
        if not t:
            return None
        lower = t.lower()

        # ---- "play <named thing>" → spotify_play ---------------------------
        # FIX: "play poker face" / "play feel good rock playlist" / "play
        # taylor swift" kept routing to media_play_pause (the LLM's
        # fallback when no better tool was obvious). The bare toggle
        # can't search — Spotify either resumed the wrong track or
        # nothing happened. Deterministic route: any "play <something>"
        # where <something> is a named entity (>= 2 chars, not a media
        # transport verb) → spotify_play(query=<something>). The Spotify
        # connector handles the search + auto-play.
        #
        # NEGATIVE filter: bare 'play' / 'play music' (transport-only)
        # AND multi-step phrases ('play X and then Y') already handled
        # elsewhere. We require the captured query to NOT be a
        # transport verb so 'play next' doesn't get sent as a search.
        _PLAY_QUERY_RE = re.compile(
            r"^(?:can\s+you\s+|could\s+you\s+|please\s+)?"
            r"play\s+(?:me\s+)?(?:some\s+|the\s+|a\s+|an\s+)?"
            r"(?P<q>.{2,80}?)"
            r"(?:\s+on\s+spotify)?[\s.!?]*$",
            re.IGNORECASE)
        _TRANSPORT_ONLY = {
            "next", "next song", "next track",
            "previous", "previous song", "previous track",
            "back", "the next song", "the next track",
            "the previous song", "the previous track",
            # "music" alone is also a transport-ish bare verb (handled
            # by media_play_pause); leave it for the LLM.
        }
        # ---- natural-language reminders ------------------------------
        # 'remind me [to X] in 5 minutes' / 'remind me at 3pm to Y' /
        # 'remind me tomorrow at 9am to Z'. Two shapes:
        #   "remind me <when> [to|that] <what>"  (when first)
        #   "remind me [to|that] <what> <when>"  (what first)
        # Both fold to a pseudo-tool dispatched to standing_orders.
        _REMIND_RE = re.compile(
            r"^(?:please\s+|can\s+you\s+|could\s+you\s+|"
            r"hey\s+iris\s+|iris\s+)?"
            r"(?:remind|nudge|ping)\s+me\s+"
            r"(?P<body>.+?)$",
            re.IGNORECASE)
        rm = _REMIND_RE.match(t)
        if rm:
            body = rm.group("body").strip().rstrip(".!?")
            # Split body into <when> + <what>.
            # Try "in N units" / "at TIME" / "tomorrow ..." prefix.
            # NOTE: longest alternatives FIRST — Python regex matches
            # left-to-right, so "m" must come AFTER "minutes"/"min"
            # or "in 5 minutes" parses as "in 5 m" + "inutes".
            when_re = re.compile(
                r"^(?P<when>(?:in\s+\d+\s*(?:seconds?|minutes?|"
                r"hours?|sec|min|hr|h|m|s))|"
                r"(?:at\s+\d{1,2}(?::\d{2})?\s*(?:am|pm)?)|"
                r"(?:tomorrow(?:\s+at\s+\d{1,2}(?::\d{2})?\s*"
                r"(?:am|pm)?)?))"
                r"\s*(?:to|that|about)?\s*(?P<what>.*)$",
                re.IGNORECASE)
            wm = when_re.match(body)
            when_text = ""
            what_text = ""
            if wm:
                when_text = wm.group("when").strip()
                what_text = wm.group("what").strip()
            else:
                # Try "[to/that] X <when>" — when is at the end.
                tail_re = re.compile(
                    r"^(?:to|that|about)?\s*(?P<what>.+?)\s+"
                    r"(?P<when>(?:in\s+\d+\s*(?:seconds?|minutes?|"
                    r"hours?|sec|min|hr|h|m|s))|"
                    r"(?:at\s+\d{1,2}(?::\d{2})?\s*(?:am|pm)?)|"
                    r"(?:tomorrow(?:\s+at\s+\d{1,2}(?::\d{2})?\s*"
                    r"(?:am|pm)?)?))$",
                    re.IGNORECASE)
                tm = tail_re.match(body)
                if tm:
                    what_text = tm.group("what").strip()
                    when_text = tm.group("when").strip()
            if when_text:
                # Normalize "at 3pm" -> "3pm" so parse_when can read it.
                when_clean = re.sub(
                    r"^at\s+", "", when_text, flags=re.IGNORECASE
                ).strip()
                return Step(
                    tool="iris_remind",
                    args={"when_text": when_clean,
                          "what": what_text or "reminder"},
                    layer="touchless",
                    description=(
                        f"remind '{what_text or 'reminder'}' "
                        f"@ {when_clean}"))

        # ---- natural-language order cancellation ---------------------
        # 'remove that reminder' / 'cancel the bacon reminder' /
        # 'delete those orders' / 'cancel all reminders'. Dispatched
        # to iris_cancel_order with args:
        #   {scope: 'all' | 'last' | 'match', query?: '<text>'}
        _CANCEL_RE = re.compile(
            r"^(?:please\s+|can\s+you\s+|could\s+you\s+|"
            r"hey\s+iris\s+|iris\s+)?"
            r"(?:cancel|remove|delete|drop|clear|kill|"
            r"unwatch|stop)\s+"
            r"(?:that|this|the|those|these|my|all|all\s+of\s+)?\s*"
            r"(?P<rest>.*)$",
            re.IGNORECASE)
        cm = _CANCEL_RE.match(t)
        if cm:
            rest = cm.group("rest").strip().rstrip(".!?")
            rl = rest.lower()
            # Heuristic: only treat as order-cancel when the verb's
            # object is clearly an order/reminder/watcher (not a
            # contact, file, tab, etc.).
            order_words = ("reminder", "reminders", "order",
                            "orders", "watcher", "watchers",
                            "watch", "alarm", "alarms", "alert",
                            "alerts", "notification",
                            "notifications", "ping", "pings",
                            "standing order", "standing orders")
            if any(w in rl for w in order_words):
                scope = "all"
                query = ""
                if "all" in rl or "every" in rl or "those" in rl \
                        or "these" in rl:
                    scope = "all"
                elif "last" in rl or "most recent" in rl:
                    scope = "last"
                else:
                    scope = "match"
                    # Strip the order-noun + 'for' to get the
                    # match query.
                    q = rest
                    for w in sorted(order_words, key=len,
                                     reverse=True):
                        q = re.sub(
                            rf"\b{re.escape(w)}\b", "",
                            q, flags=re.IGNORECASE)
                    q = re.sub(
                        r"^\s*(?:for|about|on|of)\s+", "",
                        q.strip(), flags=re.IGNORECASE)
                    query = q.strip(" .!?")
                    if not query:
                        scope = "all"
                return Step(
                    tool="iris_cancel_order",
                    args={"scope": scope, "query": query},
                    layer="touchless",
                    description=(
                        f"cancel order(s) scope={scope}"
                        + (f" query={query!r}" if query else "")))

        # ---- natural-language inbox watcher --------------------------
        # 'watch my inbox for X' / 'let me know when X arrives in my
        # inbox' / 'tell me when an email from Dani about Q3 comes'.
        _WATCH_RE = re.compile(
            r"^(?:please\s+|can\s+you\s+|could\s+you\s+|"
            r"hey\s+iris\s+|iris\s+)?"
            r"(?:watch|monitor)\s+(?:my\s+)?(?:inbox|email|mail|"
            r"emails)\s+(?:for|about)\s+(?P<q>.{2,120}?)$",
            re.IGNORECASE)
        wm = _WATCH_RE.match(t)
        if wm:
            q = wm.group("q").strip().rstrip(".!?")
            if q:
                return Step(
                    tool="iris_watch_inbox",
                    args={"query": q},
                    layer="touchless",
                    description=f"watch inbox for {q!r}")

        m = _PLAY_QUERY_RE.match(t)
        if m:
            q = m.group("q").strip().rstrip(".!?").strip()
            q_lower = q.lower()
            if q_lower and q_lower not in _TRANSPORT_ONLY:
                # IMPORTANT: SpotifyConnector is registered with
                # setup_only=True, so the registry does NOT own
                # `spotify_play` — dispatching it directly would
                # 404. Instead we route through `run_quick_command`
                # which calls the shared VoiceCommandProcessor that
                # OWNS the working Spotify path (the same one
                # `play X playlist` uses). The "on spotify" suffix
                # triggers _parse_spotify's spotify_context branch
                # so the request reliably routes to play_search_request.
                cmd = f"play {q} on spotify"
                return Step(
                    tool="run_quick_command",
                    args={"command": cmd},
                    layer="touchless",
                    description=f"play '{q}' on spotify")

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

        # ---- Google Tasks (sits ABOVE Microsoft To Do so explicit
        # 'tasks / task list / google tasks' anchored phrasing routes
        # to Google; bare 'add a task X' / 'remind me to X' falls
        # through to todo_add for Microsoft To Do). ------------------
        # tasks_add: must REQUIRE tasks-anchor word so we don't shadow
        # todo_add. Reverse forms: 'add X to my tasks', 'put X on my
        # task list', 'create task X on my google tasks'.
        _TASKS_ANCHOR = r"(?:google\s+tasks|task\s+list|to[-\s]?do\s+list|todos|to[-\s]?dos|tasks)"
        m = (re.search(
                rf"^(?:(?:please|hey\s+iris|iris)\s+)?"
                rf"(?:add|put|create|append|make)\s+"
                rf"(?:a\s+(?:new\s+)?|an\s+|new\s+)?"
                rf"(?:task|to[-\s]?do|todo|reminder|item)\s+"
                rf"(?:called\s+|titled\s+|named\s+)?"
                rf"(?P<title>.+?)\s+"
                rf"(?:to|on|in|into|onto)\s+(?:my\s+|the\s+)?{_TASKS_ANCHOR}\b\s*[.!?]*$",
                lower)
             or re.search(
                rf"^(?:(?:please|hey\s+iris|iris)\s+)?"
                rf"(?:add|put|create|append)\s+(?P<title>.+?)\s+"
                rf"(?:to|on|in|into|onto)\s+(?:my\s+|the\s+)?{_TASKS_ANCHOR}\b\s*[.!?]*$",
                lower))
        if m:
            raw_title = m.group("title").strip()
            # Strip leading articles + 'item called/titled' chrome.
            raw_title = re.sub(
                r"^(?:a|an|the|new|item\s+called|item\s+titled|"
                r"called|titled)\s+", "", raw_title,
                flags=re.IGNORECASE).strip()
            # Pull case-preserved title from original `t`.
            start = t.lower().find(raw_title)
            title = t[start:start + len(raw_title)] if start >= 0 else raw_title
            title = title.rstrip(" .,!?")
            _TASK_TITLE_STOP = {"it", "that", "this", "one", "something"}
            if title and title.lower() not in _TASK_TITLE_STOP:
                return Step(
                    tool="tasks_add",
                    args={"title": title},
                    layer="connector",
                    description=f"add Google Tasks item: {title}")

        # tasks_list: 'what's on my tasks', 'show my tasks', 'list my
        # to-dos', 'what do I have on my task list'.
        if (re.search(
                rf"^(?:(?:please|hey\s+iris|iris)\s+)?"
                rf"(?:what(?:[’'´]?s|s| is)|whats)\s+"
                rf"(?:on|in)\s+(?:my\s+|the\s+)?{_TASKS_ANCHOR}\b", lower)
            or re.search(
                rf"^(?:(?:please|hey\s+iris|iris)\s+)?"
                rf"(?:show|list|read|tell)\s+(?:me\s+)?"
                rf"(?:all\s+)?(?:my\s+|the\s+)?(?:open\s+)?{_TASKS_ANCHOR}\b",
                lower)
            or re.search(
                rf"^(?:(?:please|hey\s+iris|iris)\s+)?"
                rf"(?:what\s+do\s+i\s+have|any)\s+"
                rf"(?:on\s+)?(?:my\s+|the\s+)?{_TASKS_ANCHOR}\b", lower)
            or re.search(
                rf"^(?:(?:please|hey\s+iris|iris)\s+)?"
                rf"(?:check|do\s+i\s+have\s+any)\s+"
                rf"(?:open\s+)?(?:my\s+|the\s+)?{_TASKS_ANCHOR}\b", lower)):
            include_completed = bool(
                re.search(r"\b(?:all|completed|finished|done)\s+(?:my\s+)?"
                          rf"(?:tasks|to[-\s]?dos|todos)\b", lower))
            return Step(
                tool="tasks_list",
                args={"include_completed": include_completed},
                layer="connector",
                description="list Google Tasks")

        # tasks_complete: 'mark buy milk as done', 'check off X', 'I
        # finished X'. Capture group becomes title_match.
        m = (re.search(
                r"^(?:(?:please|hey\s+iris|iris)\s+)?"
                r"(?:mark|set|check\s+off|tick\s+off|complete|finish)\s+"
                r"(?:the\s+task\s+|my\s+task\s+)?"
                r"(?P<title>.+?)\s+"
                r"(?:as\s+)?(?:done|complete|completed|finished|off)"
                r"\s*[.!?]*$", lower)
             or re.search(
                r"^(?:(?:please|hey\s+iris|iris)\s+)?"
                r"(?:check|mark)\s+off\s+"
                r"(?P<title>.+?)\s+"
                r"(?:on|from)\s+(?:my\s+)?"
                rf"{_TASKS_ANCHOR}\s*[.!?]*$", lower))
        if m:
            raw_title = m.group("title").strip()
            raw_title = re.sub(
                r"^(?:the\s+task\s+|my\s+task\s+)", "", raw_title,
                flags=re.IGNORECASE).strip()
            start = t.lower().find(raw_title)
            title = t[start:start + len(raw_title)] if start >= 0 else raw_title
            title = title.rstrip(" .,!?")
            _TASK_PRONOUN = {"it", "that", "this", "one", "them", "all"}
            if title and title.lower() not in _TASK_PRONOUN:
                return Step(
                    tool="tasks_complete",
                    args={"title_match": title},
                    layer="connector",
                    description=f"complete Google Task: {title}")

        # tasks_delete: 'remove X from my tasks', 'delete the X task'.
        # MUST sit BELOW iris_cancel_order — and it does (cancel was
        # handled at line ~290). Requires the tasks anchor explicitly
        # so it doesn't compete with iris_forget_contact.
        m = (re.search(
                rf"^(?:(?:please|hey\s+iris|iris)\s+)?"
                rf"(?:remove|delete|drop|cancel|kill|get\s+rid\s+of)\s+"
                rf"(?P<title>.+?)\s+from\s+(?:my\s+|the\s+)?{_TASKS_ANCHOR}"
                rf"\s*[.!?]*$", lower)
             or re.search(
                rf"^(?:(?:please|hey\s+iris|iris)\s+)?"
                rf"(?:remove|delete)\s+(?:the\s+)?(?P<title>.+?)\s+task"
                rf"\s*[.!?]*$", lower)
             # Verbless-name form: 'delete called drill-task',
             # 'remove named q4-review'. User forgot to say 'task' but
             # the 'called/titled/named' anchor tells us the next token
             # is an artifact identifier — safe to route to tasks_delete
             # since Google Tasks is the primary artifact type with a
             # delete-by-name affordance. If the identifier turns out to
             # be a doc/sheet/slide instead, the connector returns a
             # clean "no such task" error and the LLM planner can retry.
             or re.search(
                rf"^(?:(?:please|hey\s+iris|iris)\s+)?"
                rf"(?:remove|delete)\s+"
                rf"(?:called|titled|named)\s+"
                rf"(?P<title>.+?)\s*[.!?]*$", lower))
        if m:
            raw_title = m.group("title").strip()
            raw_title = re.sub(
                r"^(?:the\s+|my\s+|task\s+)", "", raw_title,
                flags=re.IGNORECASE).strip()
            start = t.lower().find(raw_title)
            title = t[start:start + len(raw_title)] if start >= 0 else raw_title
            title = title.rstrip(" .,!?")
            _TASK_PRONOUN = {"it", "that", "this", "one", "them", "all", ""}
            # Ordinal / positional stop-words: "delete the last task",
            # "delete my newest task", "delete the first task" would
            # otherwise capture title='last'/'newest'/'first' and try to
            # delete a Google Task literally named that. Same treatment as
            # iris_forget_contact's _STOP set — fall through to the LLM
            # planner, which can resolve ordinals against the actual list.
            _TASK_STOPWORDS = {
                "last", "next", "first", "latest", "newest", "oldest",
                "my", "the", "a", "an", "only", "one",
                "current", "top", "bottom", "previous", "prior",
                "recent",
            }
            # Multi-word ordinals: "most recent", "highest priority",
            # "least important" — captured as multiple tokens. Bail if
            # the captured title contains any stopword AND is 2-3 words
            # (short phrases are almost always ordinal/positional refs,
            # not real task names). Longer titles like "buy milk on
            # tuesday" contain "on" but are legitimate task names.
            _title_words = (title or "").lower().split()
            _is_short_ordinal = (
                1 < len(_title_words) <= 3
                and any(w in _TASK_STOPWORDS
                        or w in {"most", "least", "highest", "lowest",
                                 "important", "priority", "urgent"}
                        for w in _title_words)
            )
            if title and title.lower() not in _TASK_PRONOUN \
                    and title.lower() not in _TASK_STOPWORDS \
                    and not _is_short_ordinal:
                return Step(
                    tool="tasks_delete",
                    args={"title_match": title},
                    layer="connector",
                    description=f"delete Google Task: {title}")

        # ---- Bare "add a task" -> Google Tasks (full CRUD available) ----
        # "add a task: buy milk" / "add task buy milk" / "remind me to
        # buy milk". Routes to tasks_add (Google) rather than todo_add
        # (MS To Do) because Google has tasks_list + tasks_complete +
        # tasks_delete — MS todo_* is add + list only, so a bare "add"
        # followed by a "delete" would break as the LLM hallucinates a
        # non-existent todo_delete tool.
        m = (re.search(r"\badd\s+(?:a\s+)?task[:\s]+(.+)$", lower)
             or re.search(r"\bremind\s+me\s+to\s+(.+)$", lower)
             or re.search(r"\bnew\s+task[:\s]+(.+)$", lower))
        if m:
            raw_title = m.group(1).strip().rstrip(".!?")
            # Strip 'called'/'titled'/'named' prefix so 'add a task called
            # drill-task' yields title='drill-task', not 'called drill-task'.
            raw_title = re.sub(
                r"^(?:called|titled|named)\s+", "", raw_title,
                flags=re.IGNORECASE).strip()
            # Recover case-preserved title from original text.
            start = t.lower().find(raw_title.lower()) if raw_title else -1
            title = t[start:start + len(raw_title)] if start >= 0 else raw_title
            title = title.rstrip(" .,!?")
            if title:
                return Step(tool="tasks_add", args={"title": title},
                            layer="connector",
                            description=f"add Google Tasks item: {title}")

        # ---- Google Forms create / responses ------------------------------
        # Placed ABOVE the gdocs/sheets/slides block so 'create a form'
        # / 'make a poll' don't get eaten by the generic doc matcher.
        # Two shapes for create:
        #   (a) inline question / question list: 'create a poll asking X'
        #   (b) numeric question count: 'make a survey with three questions
        #       about X'
        # And two shapes for read responses.
        _FORM_KIND = r"(?:poll|survey|form|quiz|questionnaire)"
        # (b) numeric count first — it's more specific and must beat the
        # broader create regex which would otherwise match without questions.
        m = re.search(
            rf"\b(?:make|create|build)\s+(?:a\s+|me\s+a\s+)?"
            rf"(?:google\s+)?{_FORM_KIND}\s+with\s+"
            r"(?P<count>\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+"
            r"questions?"
            r"(?:\s+(?:about|on|regarding|covering|for)\s+(?P<topic>.+?))?"
            r"\s*[.!?]*$",
            lower,
        )
        if m:
            count_tok = m.group("count")
            count = int(count_tok) if count_tok.isdigit() else _NUMBER_WORDS.get(count_tok, 3)
            count = max(1, min(count, 20))
            topic = (m.group("topic") or "").strip().rstrip(".!?,;")
            # Recover case-preserved topic from original text.
            if topic:
                start = t.lower().find(topic)
                if start >= 0:
                    topic = t[start:start + len(topic)]
                title = f"{topic.title()} Survey"
            else:
                title = "Untitled Form"
            questions = [
                {"title": f"Question {i + 1}", "type": "short_answer"}
                for i in range(count)
            ]
            return Step(
                tool="forms_create",
                args={"title": title, "questions": questions},
                layer="connector",
                description=f"create {count}-question Google Form")

        # (a) inline question / single-question form.
        m = re.search(
            rf"\b(?:make|create|build|new|set\s*up)\s+(?:me\s+)?"
            rf"(?:a\s+|an\s+|new\s+)?(?:google\s+)?{_FORM_KIND}\b"
            r"(?:\s+(?:titled|called|named|about|asking|on|for)\s+(?P<title>.+?))?"
            r"(?:\s+(?:with|asking|that\s+asks|to\s+ask)\s+(?P<body>.+?))?"
            r"\s*[.!?]*$",
            lower,
        )
        if m and ("with" in lower or "asking" in lower or "asks" in lower
                  or "titled" in lower or "called" in lower):
            raw_title = (m.group("title") or "").strip().rstrip(".!?,;")
            raw_body = (m.group("body") or "").strip().rstrip(".!?,;")
            # Recover case-preserved strings from original `t`.
            if raw_title:
                start = t.lower().find(raw_title)
                title = t[start:start + len(raw_title)] if start >= 0 else raw_title
            else:
                title = "Untitled Form"
            if raw_body:
                start = t.lower().find(raw_body)
                body_cased = t[start:start + len(raw_body)] if start >= 0 else raw_body
                # Numeric-prefix pattern: "three questions: A, B, C"
                num_q_m = re.match(
                    r"^\s*(\d+|two|three|four|five|six|seven|eight|nine|ten)\s+"
                    r"questions?\s*[:\-—]?\s*(?P<list>.+)$",
                    body_cased, flags=re.IGNORECASE)
                if num_q_m:
                    items = [s.strip() for s in re.split(r",|\band\b", num_q_m.group("list"))
                             if s.strip()]
                    questions = [{"title": q, "type": "short_answer"}
                                 for q in items if q]
                else:
                    # Single embedded question.
                    questions = [{"title": body_cased, "type": "short_answer"}]
            else:
                # Bare 'create a poll' with no question — synthesize a placeholder.
                questions = [{"title": "Question 1", "type": "short_answer"}]
            if questions:
                return Step(
                    tool="forms_create",
                    args={"title": title or "Untitled Form",
                          "questions": questions},
                    layer="connector",
                    description=f"create Google Form: {title}")

        # forms_responses (count-only or full read). Memory resolves the
        # form name → form_id at the orchestrator level (same convention
        # used for contact-by-name lookups); we just pass the spoken name
        # in as form_id and let the runtime resolve. When unresolvable,
        # the planner falls through to the LLM to prompt for a URL.
        m = re.search(
            rf"\b(?:how\s+many|count\s+of|number\s+of)\s+responses?\s+"
            rf"(?:on|to|for|in)\s+(?:the\s+|my\s+)?(?P<name>.+?)\s+"
            rf"{_FORM_KIND}\b",
            lower,
        )
        if m:
            name = m.group("name").strip().rstrip(".!?,;")
            return Step(
                tool="forms_responses",
                args={"form_id": name, "include_answers": False},
                layer="connector",
                description=f"count responses on form: {name}")
        m = re.search(
            rf"\b(?:show|read|check|see|view)\s+(?:me\s+)?"
            rf"(?:the\s+)?(?:responses?|answers?|results?|submissions?)\s+"
            rf"(?:on|to|for|from)\s+(?:the\s+|my\s+)?(?P<name>.+?)\s+"
            rf"{_FORM_KIND}\b",
            lower,
        )
        if m:
            name = m.group("name").strip().rstrip(".!?,;")
            return Step(
                tool="forms_responses",
                args={"form_id": name, "max": 20, "include_answers": True},
                layer="connector",
                description=f"read responses on form: {name}")

        # ---- Sheets cell-write ('add Test 1 into A2 in the Q4 plan sheet')
        # Deterministic Tier-1 so the LLM planner never has to invent a
        # spreadsheet_id; the connector resolves `sheet_name` against the
        # this-session artifact tracker (or Drive search) and writes the
        # cells. Anchored against `lower` (the regex uses [A-Z] anchored
        # to the upper-cased A1 ref from the lower-cased input — see
        # `re.IGNORECASE`). The captured values keep original case via
        # `t` so 'Test 1' stays 'Test 1' (not 'test 1').
        m = re.search(
            r"\b(?:add|put|write|set|enter|fill\s+in|type)\s+"
            r"(?P<body>.+)\s+"
            r"in\s+(?:the\s+)?(?P<name>[^.?!]+?)\s+"
            r"(?:sheet|spreadsheet)\b",
            t, flags=re.IGNORECASE,
        )
        if m:
            sheet_name = m.group("name").strip().strip("'\"")
            body = m.group("body").strip()
            # Parse "<val> in (cell )?<A1>" pairs out of body. Each
            # match becomes one cell in the values matrix.
            pair_re = re.compile(
                r"(?P<val>.+?)\s+(?:in(?:to)?|to|at)\s+(?:cell\s+)?"
                r"(?P<col>[A-Z]+)(?P<row>\d+)",
                re.IGNORECASE,
            )
            segments = re.split(r"\s+and\s+", body, flags=re.IGNORECASE)
            pairs: List[tuple[str, str, int]] = []
            for seg in segments:
                pm = pair_re.fullmatch(seg.strip())
                if not pm:
                    pairs = []
                    break
                pairs.append((
                    pm.group("val").strip().strip("'\""),
                    pm.group("col").upper(),
                    int(pm.group("row")),
                ))
            if pairs:
                def _col_idx(letters: str) -> int:
                    n = 0
                    for ch in letters:
                        n = n * 26 + (ord(ch) - ord("A") + 1)
                    return n
                min_row = min(p[2] for p in pairs)
                max_row = max(p[2] for p in pairs)
                min_col = min(_col_idx(p[1]) for p in pairs)
                max_col = max(_col_idx(p[1]) for p in pairs)
                matrix = [[""] * (max_col - min_col + 1)
                          for _ in range(max_row - min_row + 1)]
                for val, col_letters, row in pairs:
                    matrix[row - min_row][
                        _col_idx(col_letters) - min_col] = val

                def _col_label(idx: int) -> str:
                    out = ""
                    n = idx
                    while n > 0:
                        n, rem = divmod(n - 1, 26)
                        out = chr(ord("A") + rem) + out
                    return out
                if min_row == max_row and min_col == max_col:
                    a1 = f"{_col_label(min_col)}{min_row}"
                else:
                    a1 = (f"{_col_label(min_col)}{min_row}:"
                          f"{_col_label(max_col)}{max_row}")
                return Step(
                    tool="sheets_update_range",
                    args={"sheet_name": sheet_name, "range": a1,
                          "values": matrix},
                    layer="connector",
                    description=(f"write cells to {sheet_name}: {a1}"))

        # ---- Sheets cell-write, EDIT-VERB / CELL-FIRST / SHEET-FIRST ----
        # Complements the value-first parser above. Handles:
        #   'change C1 to Email'                                          (cell-first, no sheet — resolved from session artifact tracker)
        #   'set cell C1 to Email in the testing sheet'                   (cell-first + trailing sheet)
        #   'update C1 to say Email in testing'                           (with 'say' filler)
        #   'in the google sheet called testing (and) (can you) change C1 to say Email' (sheet-first)
        # Verb list intentionally kept disjoint (add|put|write|set|enter|
        # fill in|type already covered above; here we add the edit verbs
        # change|update|edit|modify|make). Match against `t` (original-
        # case) not `lower` so 'Email' stays 'Email', mirroring the
        # existing block (see the comment at classifier.py:694-697).
        _A1 = r"(?P<col>[A-Z]+)(?P<row>\d+)"
        _EDIT_VERB = r"(?:change|update|edit|modify|make|set)"
        _SAY = r"(?:say|read|equal|contain|show)"

        # (a) Sheet-first: 'in the (google )?sheet (called )?<name>[,] (and )?(can you )?<verb> (cell )?<A1> to (say )?<val>'
        m = re.search(
            rf"\bin\s+(?:the\s+)?(?:google\s+)?(?:sheet|spreadsheet)\s+"
            rf"(?:called\s+|named\s+|titled\s+)?(?P<name>[^.?!,]+?)"
            rf"[,\s]+(?:and\s+)?(?:can\s+you\s+|could\s+you\s+|please\s+)?"
            rf"{_EDIT_VERB}\s+(?:cell\s+)?{_A1}\s+to\s+(?:{_SAY}\s+)?"
            rf"(?P<val>.+?)\s*$",
            t, flags=re.IGNORECASE,
        )
        # (b) Cell-first: '<verb> (cell )?<A1> to (say )?<val> [in (the )?<name> sheet]'
        if not m:
            m = re.search(
                rf"\b{_EDIT_VERB}\s+(?:cell\s+)?{_A1}\s+to\s+(?:{_SAY}\s+)?"
                rf"(?P<val>.+?)"
                rf"(?:\s+in\s+(?:the\s+)?(?P<name>[^.?!,]+?)"
                rf"\s+(?:google\s+)?(?:sheet|spreadsheet))?"
                rf"\s*$",
                t, flags=re.IGNORECASE,
            )
        if m:
            sheet_name = (m.groupdict().get("name") or "").strip().strip("'\"")
            val = m.group("val").strip().strip("'\"").rstrip(".!?")
            a1 = f"{m.group('col').upper()}{m.group('row')}"
            return Step(
                tool="sheets_update_range",
                args={"sheet_name": sheet_name, "range": a1,
                      "values": [[val]]},
                layer="connector",
                description=(
                    f"write cell {a1} in "
                    f"{sheet_name or '(session sheet)'}"))

        # ---- Sheets cell READ / CLEAR (companions to sheets_update_range) --
        # Read:  'what is the value of cell A2', 'what's in cell A2',
        #        'read cell A2', 'get the value of A2 in the testing sheet',
        #        'show me cell A2'                        -> sheets_read_range
        # Clear: 'clear cell A2', 'empty cell A2',
        #        'remove the value in cell A2', 'delete the content in A2',
        #        'remove Email from cell A2'              -> sheets_clear_range
        # Both extract:
        #   - `range`: the A1 cell reference (upper-cased letters + digits)
        #   - `sheet_name` (optional): from 'in the sheet called X' /
        #     'in X sheet' / trailing 'in X'; when absent, the connector
        #     resolves the target from the session artifact tracker.
        # Verbs kept disjoint from the update block (add|put|write|set|
        # enter|fill in|type|change|update|edit|modify|make) so no
        # collision. Trailing sheet clause is anchored ONLY when a cell
        # reference has already matched, so a bare 'in X' can never
        # over-consume an unrelated tail. Matched against `t` (original
        # case) with IGNORECASE so a sheet name like 'Q4 Plan' keeps its
        # capitalisation.
        _A1_RC = r"(?P<col>[A-Z]+)(?P<row>\d+)"
        _TRAILING_SHEET = (
            r"(?:\s+in\s+(?:the\s+)?"
            r"(?:(?:google\s+)?(?:sheet|spreadsheet)\s+"
            r"(?:called\s+|named\s+|titled\s+))?"
            r"(?P<name>[^.?!,]+?)"
            r"(?:\s+(?:google\s+)?(?:sheet|spreadsheet))?)?"
        )

        # Leading sheet-context prefix — pulled off `t` before the read/clear
        # patterns run so 'in the sheet called testing, can you remove Email
        # from cell A2' reduces to 'can you remove Email from cell A2' with
        # the sheet name captured out-of-band. Both prefix and comma/'and'
        # bridge are optional. Only fires when a cell reference exists
        # downstream (so a bare 'in the sheet called X do something else'
        # can't hijack unrelated intents).
        _leading_sheet: str = ""
        _t_stripped = t
        if re.search(rf"\b{_A1_RC}\b", t):
            _leading_m = re.match(
                r"^(?:please\s+|hey\s+iris[,]?\s+|iris[,]?\s+)?"
                r"in\s+(?:the\s+)?"
                r"(?:(?:google\s+)?(?:sheet|spreadsheet)\s+"
                r"(?:called\s+|named\s+|titled\s+))?"
                r"(?P<lname>[^.?!,]+?)"
                r"(?:\s+(?:google\s+)?(?:sheet|spreadsheet))?"
                r"(?:\s*,\s*|\s+and\s+|\s+)",
                t, flags=re.IGNORECASE,
            )
            if _leading_m:
                _leading_sheet = (_leading_m.group("lname") or "").strip().strip("'\"")
                _t_stripped = t[_leading_m.end():].lstrip()

        # ---- READ ---------------------------------------------------------
        # (a) Interrogative: 'what is / what's [the value of|in|inside]
        #     [cell] A2 [in <sheet>]'
        m = re.search(
            rf"^(?:please\s+|can\s+you\s+|could\s+you\s+|"
            rf"hey\s+iris\s+|iris\s+)?"
            rf"what(?:\s+is|[’'´]?s)\s+"
            rf"(?:the\s+value\s+(?:of|in)\s+|in\s+|inside\s+)"
            rf"(?:cell\s+)?{_A1_RC}"
            rf"{_TRAILING_SHEET}\s*[.?!]*$",
            _t_stripped, flags=re.IGNORECASE,
        )
        # (b) Imperative: 'read / show me / get [the value of] [cell] A2
        #     [in <sheet>]'
        if not m:
            m = re.search(
                rf"^(?:please\s+|can\s+you\s+|could\s+you\s+|"
                rf"hey\s+iris\s+|iris\s+)?"
                rf"(?:"
                rf"read\s+(?:the\s+value\s+(?:of|in)\s+)?"
                rf"|show\s+(?:me\s+)?(?:the\s+value\s+(?:of|in)\s+)?"
                rf"|get\s+(?:me\s+)?(?:the\s+value\s+(?:of|in)\s+)?"
                rf"|tell\s+me\s+(?:what(?:\s+is|[’'´]?s)\s+"
                rf"(?:the\s+value\s+(?:of|in)\s+|in\s+)?)?"
                rf")(?:cell\s+)?{_A1_RC}"
                rf"{_TRAILING_SHEET}\s*[.?!]*$",
                _t_stripped, flags=re.IGNORECASE,
            )
        if m:
            sheet_name = (m.groupdict().get("name") or "").strip().strip("'\"")
            if not sheet_name and _leading_sheet:
                sheet_name = _leading_sheet
            a1 = f"{m.group('col').upper()}{m.group('row')}"
            args_read: Dict[str, Any] = {"range": a1}
            if sheet_name:
                args_read["sheet_name"] = sheet_name
            return Step(
                tool="sheets_read_range",
                args=args_read,
                layer="connector",
                description=(
                    f"read cell {a1} in "
                    f"{sheet_name or '(session sheet)'}"))

        # ---- CLEAR --------------------------------------------------------
        # (a) Bare clear/empty: 'clear cell A2', 'empty cell A2',
        #     'blank out A2 in <sheet>'
        m = re.search(
            rf"^(?:please\s+|can\s+you\s+|could\s+you\s+|"
            rf"hey\s+iris\s+|iris\s+)?"
            rf"(?:clear|empty|blank)\s+(?:out\s+)?(?:cell\s+)?{_A1_RC}"
            rf"{_TRAILING_SHEET}\s*[.?!]*$",
            _t_stripped, flags=re.IGNORECASE,
        )
        # (b) Remove/delete with intermediate phrase:
        #     'remove the value|content|contents|text|entry in|from|of
        #      [cell] A2'
        #     'delete the content in A2'
        #     'remove Email from cell A2'   (value being removed = 'Email')
        # The alternation is tried in this order:
        #   1. 'the <value-noun> (in|from|of)' — matches the canonical
        #      'remove the value in cell A2' form.
        #   2. '<anything> from' — matches 'remove Email from cell A2'
        #      where the token before 'from' is the current cell content.
        # We DON'T capture the removed value: the tool just clears the
        # cell (semantically 'make empty'), so the noun preceding 'from'
        # is only used to disambiguate that this is a cell-clear intent,
        # not a memory-forget intent.
        if not m:
            m = re.search(
                rf"^(?:please\s+|can\s+you\s+|could\s+you\s+|"
                rf"hey\s+iris\s+|iris\s+)?"
                rf"(?:remove|delete|erase|wipe)\s+"
                rf"(?:"
                rf"the\s+(?:value|content|contents|text|entry|data)"
                rf"\s+(?:in|from|of)\s+"
                rf"|[^.?!,]+?\s+from\s+"
                rf")"
                rf"(?:cell\s+)?{_A1_RC}"
                rf"{_TRAILING_SHEET}\s*[.?!]*$",
                _t_stripped, flags=re.IGNORECASE,
            )
        if m:
            sheet_name = (m.groupdict().get("name") or "").strip().strip("'\"")
            if not sheet_name and _leading_sheet:
                sheet_name = _leading_sheet
            a1 = f"{m.group('col').upper()}{m.group('row')}"
            args_clear: Dict[str, Any] = {"range": a1}
            if sheet_name:
                args_clear["sheet_name"] = sheet_name
            return Step(
                tool="sheets_clear_range",
                args=args_clear,
                layer="connector",
                description=(
                    f"clear cell {a1} in "
                    f"{sheet_name or '(session sheet)'}"))

        # ---- Google Docs / Sheets / Slides create -------------------------
        # Title capture stops at common chain words ("and write/include/add/
        # fill/put/...", "then", commas) so requests like 'create doc called
        # X AND write Y' don't grab the whole tail as the title — those
        # chain forms should yield to multi-action / Tier 2.
        _TITLE_STOP = (
            r"\s+(?:"
            r"and\s+(?:write|include|add|put|insert)|"
            r"and\s+(?:the\s+body\s+should\s+say|have\s+it\s+say|"
            r"make\s+it\s+say)|"
            r"then|,|;|"
            r"saying|reading|"
            r"that\s+(?:says|reads|contains|has)|"
            r"which\s+(?:says|reads|contains|has)|"
            r"containing|"
            r"with\s+(?:the\s+)?(?:first\s+slide|title\s+slide|"
            r"opening\s+slide|heading|"
            r"header(?:s)?|column(?:s)?|(?:first\s+)?row(?:s)?|"
            r"data|values?|"
            r"weather|email|content|contents|"
            r"text|body|"
            r"morning|debrief|summary|briefing|brief)|"
            r"with\s+(?:the\s+)?(?:body|text|content|contents)\s+of|"
            r"(?:[—\-:]|\.)\s*(?:body|content|text)\s*[:\-—]|"
            r"with\s+(?=\S+(?:\s+\S+){0,40}?\s+"
            r"(?:written|inside|in\s+it|in\s+there|"
            r"in\s+the\s+(?:doc(?:ument)?|sheet|slide(?:show)?|"
            r"presentation|deck|file))\b)|"
            r"with\s+(?=\S+(?:\s+\S+){0,40}?\s+(?:as|for)\s+"
            r"(?:the\s+)?(?:body|content|contents|text)\b)"
            r")\b"
        )
        _BODY_PREFIX = (
            r"(?:with\s+(?:the\s+)?(?:body|text|content|contents)"
            r"(?:\s+of)?(?:\s+(?:reading|saying))?\s+"
            r"|with\s+(?:the\s+)?first\s+slide(?:\s+(?:saying|that\s+says|reading|titled))?\s+"
            r"|with\s+(?:the\s+)?title\s+slide(?:\s+(?:saying|that\s+says|reading))?\s+"
            r"|with\s+(?:the\s+)?opening\s+slide(?:\s+(?:saying|that\s+says))?\s+"
            r"|with\s+(?:the\s+)?heading\s+"
            r"|with\s+(?:the\s+)?header(?:s)?\s+"
            r"|with\s+(?:the\s+)?column(?:s)?\s+"
            r"|with\s+(?:the\s+)?(?:first\s+)?row(?:s)?\s+"
            r"|with\s+(?:the\s+)?data\s+"
            r"|with\s+(?:the\s+)?values?\s+"
            r"|saying\s+|reading\s+"
            r"|that\s+(?:says|reads|contains|has)\s+"
            r"(?:the\s+(?:text|words?|phrase|line)\s+)?"
            r"|which\s+(?:says|reads|contains|has)\s+"
            r"(?:the\s+(?:text|words?|phrase|line)\s+)?"
            r"|containing\s+"
            r"|and\s+(?:write|include|add|put|insert)\s+"
            r"(?:the\s+(?:text|words?|phrase|line)\s+)?"
            r"|and\s+(?:the\s+body\s+should\s+say|have\s+it\s+say|"
            r"make\s+it\s+say)\s+"
            r"|(?:[—\-:]|\.\s+)\s*(?:body|content|text)\s*[:\-—]\s*"
            r"|with\s+)"
        )

        # ---- Slides: ADD a slide to an EXISTING presentation ----------
        # MUST sit ABOVE the generic slides_create block below (line ~1037)
        # so 'new slide in <deck>' / 'create another slide in <deck>' don't
        # hijack to a fresh empty presentation. Matches:
        #   'add a slide to <deck>'
        #   'add a slide to <deck> titled <title>'
        #   'add a slide titled <title> to <deck>'
        #   'put another slide in <deck>'
        #   'new slide in <deck> saying <body>'
        #   'add a slide titled <title> saying <body> to <deck>'
        # `presentation_name` is passed spoken; the connector must resolve it
        # against the google_picker_cache (populated at slides_create time,
        # kind='slide') the same way sheets_connector._resolve_sid_by_name
        # resolves sheet_name -> spreadsheet_id. Until slides_connector grows
        # that resolver + accepts presentation_name in its schema, this route
        # will error at connector-dispatch time — pair this classifier edit
        # with a matching slides_connector patch.
        _SLIDE_ADD_VERB = r"(?:add|put|append|insert|create|make|new|stick|throw)"
        _SLIDE_TO_IN = r"(?:to|in|into|onto|on)"
        _SLIDE_TITLED = r"(?:titled|called|named|entitled|with\s+(?:the\s+)?title)"
        _SLIDE_SAYING = (
            r"(?:saying|reading|containing|that\s+says|which\s+says|"
            r"with\s+(?:the\s+)?(?:body|content|contents|text)|"
            r"with\s+body|with)"
        )
        _SLIDE_DECK_NOUN = (
            r"(?:(?:google\s+)?(?:presentation|deck|slideshow|slide\s+deck|"
            r"slides))"
        )

        # Form A: <verb> [a|another|new] slide [titled X] [saying Y] <to> <deck>
        m = re.search(
            rf"^(?:please\s+|can\s+you\s+|could\s+you\s+|"
            rf"hey\s+iris\s+|iris\s+)?"
            rf"{_SLIDE_ADD_VERB}\s+(?:a\s+|another\s+|one\s+more\s+|"
            rf"a\s+new\s+|another\s+new\s+|new\s+)?"
            rf"slide\s+"
            rf"(?:{_SLIDE_TITLED}\s+(?P<title>.+?)\s+)?"
            rf"(?:{_SLIDE_SAYING}\s+(?P<body>.+?)\s+)?"
            rf"{_SLIDE_TO_IN}\s+(?:my\s+|the\s+)?"
            rf"(?:{_SLIDE_DECK_NOUN}\s+(?:called\s+|named\s+|titled\s+)?)?"
            rf"(?P<pres>.+?)"
            rf"(?:\s+{_SLIDE_DECK_NOUN})?"
            rf"\s*[.!?]*$",
            lower,
        )
        # Form B: <verb> [a|another|new] slide <to> <deck> [titled X] [saying Y]
        if not m:
            m = re.search(
                rf"^(?:please\s+|can\s+you\s+|could\s+you\s+|"
                rf"hey\s+iris\s+|iris\s+)?"
                rf"{_SLIDE_ADD_VERB}\s+(?:a\s+|another\s+|one\s+more\s+|"
                rf"a\s+new\s+|another\s+new\s+|new\s+)?"
                rf"slide\s+"
                rf"{_SLIDE_TO_IN}\s+(?:my\s+|the\s+)?"
                rf"(?:{_SLIDE_DECK_NOUN}\s+(?:called\s+|named\s+|titled\s+)?)?"
                rf"(?P<pres>.+?)"
                rf"(?:\s+{_SLIDE_DECK_NOUN})?"
                rf"(?:\s+{_SLIDE_TITLED}\s+(?P<title>.+?))?"
                rf"(?:\s+{_SLIDE_SAYING}\s+(?P<body>.+?))?"
                rf"\s*[.!?]*$",
                lower,
            )
        if m:
            def _cased(field: str) -> str:
                raw = (m.groupdict().get(field) or "").strip().strip("'\"")
                raw = raw.rstrip(".!?,;")
                if not raw:
                    return ""
                idx = t.lower().find(raw)
                return t[idx:idx + len(raw)] if idx >= 0 else raw
            pres_name = _cased("pres")
            title = _cased("title") or "Untitled slide"
            body = _cased("body")
            # Guard: presentation name must be a real noun, not a pronoun
            # (prevents 'add a slide to it' from firing before the LLM
            # can resolve the referent) and must not be a bare deck-noun.
            _PRES_STOP = {
                "it", "that", "this", "one", "them", "here", "there",
                "presentation", "deck", "slideshow", "slides",
                "slide deck", "the presentation", "the deck", "",
            }
            if pres_name and pres_name.lower() not in _PRES_STOP:
                args_add: Dict[str, Any] = {
                    "presentation_name": pres_name,
                    "title": title,
                }
                if body:
                    args_add["body"] = body
                return Step(
                    tool="slides_add_slide",
                    args=args_add,
                    layer="connector",
                    description=(
                        f"add slide '{title}' to presentation '{pres_name}'"))

        m = re.search(
            r"\b(?:make|create|new)\s+(?:a\s+)?(?:new\s+)?(?:google\s+)?"
            r"(doc(?:ument)?|sheet|spreadsheet|slides?(?:how)?|presentation|deck)\b"
            r"(?:\s+(?:titled|called|named|title|name)\s+(.+?))?"
            rf"(?:{_TITLE_STOP}\s*(?P<body>.+?)\s*$|\s*$)",
            lower,
        )
        if m:
            kind = m.group(1)
            title = (m.group(2) or "Untitled").strip().rstrip(".!?").strip("'\"")
            body = ""
            try:
                start = m.start("body")
                end = m.end("body")
            except (IndexError, re.error):
                start = end = -1
            if start >= 0 and end > start:
                raw_tail_cased = t[start:end] if end <= len(t) else (m.group("body") or "")
                body = re.sub(rf"^{_BODY_PREFIX}", "", raw_tail_cased,
                              flags=re.IGNORECASE)
                body = body.strip().strip("'\"").rstrip(".!?").strip()
                body = re.sub(r"\s+(?:written\s+)?in\s+(?:it|there|"
                              r"the\s+doc(?:ument)?|"
                              r"the\s+sheet|the\s+slides?(?:how)?|"
                              r"the\s+deck|the\s+presentation|the\s+file|"
                              r"the\s+first\s+slide|the\s+title\s+slide)\s*$",
                              "", body, flags=re.IGNORECASE).strip()
                body = re.sub(r"\s+(?:as|for)\s+(?:the\s+)?"
                              r"(?:body|content|contents|text)\s*$",
                              "", body, flags=re.IGNORECASE).strip()
                body = re.sub(r"\s+(?:inside|written)\s*$", "", body,
                              flags=re.IGNORECASE).strip()
            if kind.startswith("doc"):
                doc_args: Dict[str, Any] = {"title": title}
                if body:
                    doc_args["text"] = body
                return Step(tool="gdocs_create", args=doc_args, layer="connector",
                            description=f"create Google Doc: {title}")
            if kind in ("sheet", "spreadsheet"):
                sheet_args: Dict[str, Any] = {"title": title}
                if body:
                    rows = _parse_sheet_body(body)
                    if rows:
                        sheet_args["rows"] = rows
                return Step(tool="sheets_create", args=sheet_args, layer="connector",
                            description=f"create Google Sheet: {title}")
            slides_args: Dict[str, Any] = {"title": title}
            if body:
                slides_args["heading"] = body
            return Step(tool="slides_create", args=slides_args, layer="connector",
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

        # ---- Google Photos upload (explicit local path) -------------------
        # Mirrors drive_upload but targets the Photos library. Matched
        # against original-case `t` so Windows paths keep their drive
        # letter / casing.
        m = re.search(
            r"\bupload\s+(?P<path>[^\s\"]+|\"[^\"]+\"|'[^']+')\s+to\s+"
            r"(?:my\s+)?(?:google\s+)?photos\b",
            t, flags=re.IGNORECASE,
        )
        if m:
            path = m.group("path").strip("'\"")
            return Step(tool="photos_upload", args={"path": path},
                        layer="connector",
                        description=f"upload to Google Photos: {path}")

        # Save-screenshot / image phrasing — no explicit path. We pass a
        # symbolic placeholder so the orchestrator can resolve it
        # (last_screenshot / latest file). When unresolvable, the runtime
        # falls through to the LLM which can pick the right path via
        # list_files.
        m = re.search(
            r"\b(?:save|put|add|send|stick)\s+"
            r"(?:this|that|the|my\s+latest)\s+"
            r"(?P<kind>screenshot|screen\s*shot|photo|picture|image|pic|"
            r"video|clip|recording)\s+"
            r"(?:in|to|into|onto)\s+(?:my\s+)?(?:google\s+)?photos\b",
            lower,
        )
        if m:
            kind = m.group("kind").strip()
            return Step(
                tool="photos_upload",
                args={"path": "{last_screenshot}"},
                layer="connector",
                description=f"upload latest {kind} to Google Photos")

        # ---- Email compose (recipient + body, simple phrasing) -----------
        # Pre-guard (defensive): skip BOTH email-compose regexes below when
        # the utterance targets a spreadsheet cell edit or a Docs/Slides/
        # OneNote document. Prevents "...change C1 to say Email" (where
        # "Email" is CONTENT, not a compose verb) from ever slipping past
        # a future regex regression into outlook_compose. Falls through to
        # the sheets_update_range / gdocs_* / slides_* patterns that own
        # those utterances.
        _EMAIL_SKIP_CONTEXT = (
            "sheet", "spreadsheet", "google doc", "document",
            "slide", " cell ", " range ", " row ", " column ",
        )
        _EMAIL_A1_RE = re.compile(r"\b[A-Z]{1,3}[0-9]{1,4}\b")
        _skip_email_compose = (
            _EMAIL_A1_RE.search(t) is not None
            or any(k in lower for k in _EMAIL_SKIP_CONTEXT)
        )

        # Recognized forms. Verb-aware routing: an explicit "send" verb
        # produces an email_send Step (opens the draft AND presses the mail
        # app's send shortcut); the bare "email X saying Y" phrasing reads
        # as "open a draft" and produces outlook_compose (which the
        # orchestrator MAY still rewrite to gmail_send/ms_mail_send based
        # on the user's default_send_via preference and resolve a name →
        # email from memory if recipient isn't an @ address):
        #
        #   "email <addr> saying <body>"                 → outlook_compose
        #   "email <Name> saying <body>"                 → outlook_compose
        #   "email to <addr> saying <body>"              → outlook_compose
        #   "send (me|a|an) email (to) <addr|Name> saying <body>"
        #                                                → email_send
        #   "send a message to <addr|Name> saying <body>"→ email_send
        # Excluded: pronouns/articles ("me", "him", "the boss" — too
        # ambiguous; let those fall through to higher tiers).
        m = None if _skip_email_compose else re.search(
            r"\b(?P<sendverb>send\s+(?:me\s+)?(?:an?\s+)?)?"
            r"(?:e[-\s]?mail|message)\s+"
            r"(?:to\s+)?"
            r"(?P<to>\S+@\S+\.\S+|[A-Za-z][A-Za-z0-9._\-]*)"
            r"\s+(?:saying|with\s+(?:the\s+)?message|that\s+says|that|with)\s+"
            r"(?P<body>.+)$",
            t, flags=re.IGNORECASE,
        )
        if m:
            recipient = m.group("to").strip(",.;")
            # Reject ambiguous "recipients" so we don't grab the wrong word.
            _STOP_NAMES = {"me", "him", "her", "them", "us", "the", "a", "an"}
            if "@" in recipient or recipient.lower() not in _STOP_NAMES:
                if m.group("sendverb"):
                    # Explicit "send" verb → send tool with confirm gate
                    # forced on the Step (not just via RISKY_TOOLS). Tool
                    # stays email_send so the orchestrator's rewrite path
                    # can still upgrade to gmail_send / ms_mail_send when
                    # the user has set default_send_via.
                    tool = "email_send"
                    description = "send email"
                    needs_confirm = True
                else:
                    tool = "outlook_compose"
                    description = "compose email draft"
                    needs_confirm = False
                return Step(
                    tool=tool,
                    args={"recipient": recipient,
                          "body": m.group("body").strip()},
                    layer="connector",
                    description=description,
                    needs_confirm=needs_confirm,
                )

        # ---- Email compose, VERBLESS form --------------------------------
        # "email <addr|Name> <body...>" without saying/that/about. Runs ONLY
        # after the connective form above failed, so "email vesko saying hi"
        # still hits the richer parser. Same recipient rules (single token,
        # @addr or letters/'/-). Body strips surrounding quotes so
        #   email vesko "hello world"  → body="hello world"
        # Limitation: multi-word names take only the first token as recipient
        # ("email konstantin markov hello world" → recipient="konstantin",
        # body="markov hello world"). Users with multi-word contact names
        # should use the connective form ("email konstantin markov saying
        # hello world").
        m = None if _skip_email_compose else re.search(
            r"^(?:please\s+|hey\s+)?"
            r"(?P<sendverb>send\s+(?:me\s+)?(?:an?\s+)?)?"
            r"(?:e[-\s]?mail|mail)\s+(?:to\s+)?"
            r"(?P<to>\S+@\S+\.\S+|[A-Za-z][A-Za-z'\-]+)"
            r"\s+(?P<body>\S(?:.*\S)?)\s*$",
            t, flags=re.IGNORECASE,
        )
        if m:
            recipient = m.group("to").strip(",.;:!?")
            body = m.group("body").strip().strip('"').strip("'").strip()
            _STOP_NAMES = {"me", "myself", "him", "her", "them", "us",
                           "everyone", "everybody", "anyone", "anybody",
                           "someone", "somebody",
                           "the", "a", "an", "my", "your", "our", "their", "all"}
            _DEFERRAL_BODIES = {"later", "tomorrow", "in a bit", "soon"}
            if (("@" in recipient or recipient.lower() not in _STOP_NAMES)
                    and len(body) >= 2
                    and body.lower() not in _DEFERRAL_BODIES):
                # Verb-aware routing: explicit "send" verb → email_send
                # (opens draft + presses Ctrl+Enter); bare "email X body"
                # → outlook_compose (draft only). Send-verb path forces
                # needs_confirm=True on the Step (belt+suspenders with
                # RISKY_TOOLS).
                if m.group("sendverb"):
                    tool = "email_send"
                    description = "send email (verbless)"
                    needs_confirm = True
                else:
                    tool = "outlook_compose"
                    description = "compose email draft (verbless)"
                    needs_confirm = False
                return Step(
                    tool=tool,
                    args={"recipient": recipient, "body": body},
                    layer="connector",
                    description=description,
                    needs_confirm=needs_confirm,
                )

        # ---- text generation -> local Ollama (free, instant, no realtime
        # tokens / no OpenAI rate limit). Placed HERE (not at end of classify)
        # so the regex-gen pattern wins over iris_lookup_contact's loose
        # "give me ... email ..." matcher; the patterns are tight enough that
        # they don't pre-empt earlier specific intents (volume/email/etc.).
        # If Ollama isn't installed/running, handles_connector returns False
        # and these steps fall through to realtime — identical to behavior
        # before this block existed.

        # (0a) Explicit prefix: 'use ollama / local model to X', 'use the
        # local llm: X'. User EXPLICITLY asked for the cheap path — strip
        # the routing prefix and feed the rest as the prompt.
        m = re.search(
            r"^\s*(?:please\s+|hey\s+)?"
            r"(?:use|using|via|with)\s+(?:the\s+)?"
            r"(?:ollama|local\s+model|local\s+llm|local\s+ai|"
            r"cheap\s+model|cheap\s+path)\s*"
            r"(?:to\s+|,\s*|:\s*)?"
            r"(?P<task>.+)",
            lower,
        )
        if m and len(m.group("task").strip()) >= 3:
            # Pull from ORIGINAL text (case-preserving) by cutting the same
            # number of leading chars `lower` consumed.
            task = t[len(t) - len(m.group("task")):].strip()
            return Step(
                tool="ollama_generate", args={"prompt": task},
                layer="connector",
                description="explicit local-model route")
        # (0b) Suffix form: 'X via ollama' / 'X with the local model'.
        m = re.search(
            r"^(?P<task>.+?)\s+"
            r"(?:via|with|using)\s+(?:the\s+)?"
            r"(?:ollama|local\s+model|local\s+llm|local\s+ai|cheap\s+model)"
            r"\b[\s.!?]*$",
            lower,
        )
        if m and len(m.group("task").strip()) >= 3:
            task = t[: len(m.group("task"))].strip().rstrip(",.;:!?")
            return Step(
                tool="ollama_generate", args={"prompt": task},
                layer="connector",
                description="explicit local-model route")

        # (1) Creative-write: 'write/give/compose a haiku/poem/tweet/joke
        # about X'. Restricted to clearly creative text snippets so 'write
        # code/a function/a script' falls through to realtime (which has
        # codebase context), and 'create google doc' / 'compose email Dani
        # saying hi' have already won above.
        m = re.search(
            r"^\s*(?:please\s+|hey\s+|can\s+you\s+|could\s+you\s+|"
            r"will\s+you\s+)*"
            r"(?:write|compose|draft|generate|create|make|give)\s+"
            r"(?:me\s+|us\s+|out\s+)?(?:a|an|some|the|few)?\s*"
            r"(?:short\s+|quick\s+|funny\s+|brief\s+|long\s+|silly\s+|"
            r"serious\s+|clever\s+|nice\s+|good\s+|cool\s+|catchy\s+)?"
            r"(?P<kind>haikus?|poems?|limericks?|sonnets?|verses?|rhymes?|"
            r"sentences?|paragraphs?|essays?|fables?|jokes?|"
            r"riddles?|tweets?|posts?|captions?|descriptions?|bios?|"
            r"taglines?|slogans?|mottos?|quotes?|sayings?|puns?|"
            r"(?:story|stories))"
            r"\b",
            lower,
        )
        if m:
            return Step(
                tool="ollama_generate", args={"prompt": t},
                layer="connector",
                description=f"generate {m.group('kind')} locally")

        # (2) Regex generation: 'write/give me a regex for X', 'regex that
        # matches X'. Self-contained — the spec is always inline. MUST sit
        # above iris_lookup_contact, whose loose '(give me)...email' matcher
        # would otherwise steal 'give me a regex for email addresses'.
        if (re.search(
                r"^\s*(?:please\s+|hey\s+)?(?:write|give|make|generate)\s+"
                r"(?:me\s+)?(?:a|an)?\s*regex\b", lower)
            or re.search(
                r"^\s*regex\s+(?:for|that|to|matching)\b", lower)):
            return Step(
                tool="ollama_generate", args={"prompt": t},
                layer="connector",
                description="regex via local model")

        # (3) Translate with inline source ('translate X to Spanish',
        # 'translate to French: X'). Excludes 'translate this/that/the
        # screen' which need screen context — fall through to realtime.
        if (re.search(
                r"^\s*(?:please\s+)?translate\s+"
                r"(?!this\b|that\b|it\b|"
                r"the\s+(?:screen|page|document|email|message|text|"
                r"selection|highlighted)\b)"
                r".{3,}\s+(?:to|into)\s+\w+",
                lower)
            or re.search(
                r"^\s*(?:please\s+)?translate\s+(?:to|into)\s+\w+"
                r"[\s:,]+.{3,}",
                lower)):
            return Step(
                tool="ollama_generate", args={"prompt": t},
                layer="connector",
                description="translate locally")

        # (4) Paraphrase / rephrase / proofread with inline source (quoted
        # or colon-delimited). 'rephrase that' is excluded — needs context.
        if re.search(
                r"^\s*(?:please\s+)?(?:rephrase|reword|paraphrase|"
                r"proofread|fix\s+(?:the\s+)?grammar\s+(?:of|in)|"
                r"correct\s+(?:the\s+)?grammar\s+(?:of|in))"
                r"\s*[:\"'].+",
                lower):
            return Step(
                tool="ollama_generate", args={"prompt": t},
                layer="connector",
                description="rewrite locally")

        # ---- weather ("what's the weather", "is it raining", "weather in X")
        # Free, no-key wttr.in lookup. Auto-detects location when none given.
        #
        # Order matters: TEMPORAL-phrase patterns are tried FIRST so they
        # win over the "weather in <place>" pattern, which would otherwise
        # greedily capture "the next three days" / "tomorrow" / "this week"
        # as a location and then fail to geocode them. All temporal /
        # generic phrasings route to weather_get with NO location (empty
        # string → IP auto-detect), and weather.py's casual `summary`
        # field becomes the user-facing reply via orchestrator._format_message.
        #
        # (a) Temporal-phrase forecast requests: "forecast for the next
        #     three days", "weather for tomorrow", "forecast for this week",
        #     "weather for the weekend", "forecast for today", etc.
        #     These would otherwise be eaten by the location pattern below.
        if re.search(
            r"\b(?:weather|forecast|temperature|temp|conditions?)\s+"
            r"(?:in|for|over|during|across|across\s+the|through|throughout)\s+"
            r"(?:the\s+)?"
            r"(?:next\s+(?:\d+|few|couple\s+of|several|two|three|four|five|six|seven|"
            r"ten|fourteen|24|48|72)\s+"
            r"(?:hour|day|night|week|weekend|morning|evening|afternoon)s?|"
            r"today|tonight|tomorrow|yesterday|"
            r"this\s+(?:morning|afternoon|evening|night|week|weekend)|"
            r"the\s+(?:morning|afternoon|evening|night|week|weekend|day|coming\s+days)|"
            r"(?:mon|tues|wednes|thurs|fri|satur|sun)day)"
            r"\b",
            t, flags=re.IGNORECASE,
        ):
            args: Dict[str, Any] = {}
            days = _extract_forecast_days(t)
            if days:
                args["days"] = days
            return Step(
                tool="weather_get",
                args=args,
                layer="touchless",
                description="weather forecast (auto-detect location)",
            )
        # (a.2) "[N] day forecast" / "[N]-day forecast" / "4 day weather" —
        # the prepositionless form. "show me the forecast", "give me the
        # forecast", "today's forecast" also land here. All deterministic.
        if re.search(
            r"\b(?:"
            # "4 day forecast", "four day forecast", "4-day forecast"
            r"(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten)"
            r"[-\s]+day[-\s]+(?:forecast|weather|outlook)|"
            # "forecast for today", "today's forecast", "todays forecast"
            r"(?:today(?:[’'´]?s)?|tomorrow(?:[’'´]?s)?|tonight(?:[’'´]?s)?|"
            r"this\s+(?:week|weekend|morning|evening|afternoon|night)|"
            r"the\s+(?:week|weekend))"
            r"[\s'’]*forecast|"
            # "show / give / tell me the forecast"
            r"(?:show|give|tell|read)\s+me\s+(?:the\s+)?"
            r"(?:forecast|weather|temperature|temp|conditions?)|"
            # bare "forecast" / "the forecast" at start of utterance
            r"^(?:the\s+)?(?:weather\s+)?forecast\??$|"
            # "weather outlook", "5-day outlook"
            r"weather\s+outlook"
            r")\b",
            t, flags=re.IGNORECASE,
        ):
            args = {}
            days = _extract_forecast_days(t)
            if days:
                args["days"] = days
            return Step(
                tool="weather_get",
                args=args,
                layer="touchless",
                description="weather forecast (auto-detect location)",
            )
        # (b) "weather in <place>" / "weather for <place>" / "forecast in X".
        # Negative lookahead rejects temporal phrases (today/tomorrow/next N
        # days/this week/etc.) so they fall through to the (a) branch above
        # rather than being mis-captured as a place name.
        m = re.search(
            r"\b(?:weather|forecast|temperature|temp)\s+"
            r"(?:in|for|at|of|near|around)\s+"
            r"(?!(?:the\s+)?(?:next|today|tonight|tomorrow|yesterday|"
            r"this|the\s+(?:morning|afternoon|evening|night|week|weekend|day|coming))\b)"
            r"(?P<loc>[A-Za-z][A-Za-z0-9 ,.\-'’]{1,80})\??$",
            t, flags=re.IGNORECASE,
        )
        if m:
            return Step(
                tool="weather_get",
                args={"location": m.group("loc").strip(" ?.,")},
                layer="touchless",
                description=f"weather in {m.group('loc').strip()}",
            )
        # (c) "what's/how's the weather", "is it raining/snowing/hot/cold",
        # "what's the temperature", "should I bring an umbrella",
        # "will it rain", "how hot/cold/warm is it", "any rain today".
        m = re.search(
            r"\b("
            r"(?:what(?:[’'´]?s|s'?s| is)|how(?:[’'´]?s| is)|tell\s+me|"
            r"give\s+me|show\s+me)\s+(?:about\s+)?(?:the\s+)?"
            r"(?:weather|forecast|temperature|temp|conditions?|"
            r"high|low|highs?\s+and\s+lows?)|"
            r"(?:what(?:'s|s| is)|how)\s+(?:the\s+)?"
            r"(?:weather|forecast|temperature|temp)\s+(?:like|looking|looks)|"
            r"is\s+it\s+(?:raining|snowing|cold|hot|warm|chilly|nice\s+out|"
            r"freezing|sunny|cloudy|humid|windy|"
            r"going\s+to\s+(?:rain|snow|storm|be\s+(?:cold|hot|warm|nice|sunny|cloudy)))|"
            r"will\s+it\s+(?:rain|snow|storm|be\s+(?:cold|hot|warm|nice|sunny|cloudy|"
            r"raining|snowing))|"
            r"(?:gonna|going\s+to)\s+(?:rain|snow|storm)|"
            r"how\s+(?:hot|cold|warm|chilly|humid|windy)\s+(?:is\s+it|will\s+it\s+be)|"
            r"any\s+(?:rain|snow|storms?)\s+(?:today|tonight|tomorrow|this\s+week)|"
            r"chance\s+of\s+(?:rain|snow|storms?|precipitation)|"
            r"(?:do\s+i\s+need|should\s+i\s+(?:bring|wear|grab))\s+"
            r"(?:an?\s+)?(?:umbrella|jacket|coat|sweater|raincoat|boots|"
            r"sunscreen|sunglasses)"
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

        # ---- Google identity (whoami + birthday) --------------------------
        # MUST sit ABOVE the email-reads block — 'what's my email' (the
        # whoami intent) would otherwise be stolen by email-summary's
        # `(emails?|...)` alternation. Tight anchored patterns; no LLM.
        if re.search(
            r"^\s*(?:please\s+|hey\s+iris\s+|iris\s+)?"
            r"(?:who\s+am\s+i"
            r"(?:\s+(?:signed|logged|currently\s+signed|currently\s+logged)\s+"
            r"(?:in|on)(?:\s+as)?)?"
            r"|what(?:[’'´]?s|s| is)\s+my\s+"
            r"(?:google\s+|primary\s+)?"
            r"(?:email(?:\s+address)?|account|name|profile|"
            r"user(?:name)?\s+on\s+google)"
            r"|what\s+account\s+(?:am\s+i\s+(?:on|using)|"
            r"is\s+(?:this|signed\s+in)))"
            r"\s*\??\s*$",
            lower,
        ):
            return Step(
                tool="google_whoami",
                args={},
                layer="connector",
                description="who am I signed in as (Google)")
        if re.search(
            r"^\s*(?:please\s+|hey\s+iris\s+|iris\s+)?"
            r"(?:when(?:[’'´]?s|s| is)\s+my\s+(?:birthday|bday|b[-\s]?day)"
            r"|what(?:[’'´]?s|s| is)\s+my\s+"
            r"(?:birthday|bday|date\s+of\s+birth|dob)"
            r"|how\s+old\s+am\s+i"
            r"|when\s+do\s+i\s+turn\s+\d+)"
            r"\s*\??\s*$",
            lower,
        ):
            return Step(
                tool="google_my_birthday",
                args={},
                layer="connector",
                description="my Google profile birthday")

        # ---- email reads ("summarize my unread emails", "what's in my
        # inbox", "any new mail", "show me my unread", "read my emails")
        # -----------------------------------------------------------------
        # Routes through Layer 1 so the connector's deterministic
        # `summary` field is what the user sees — bypassing the LLM,
        # which has been caught fabricating demo emails when handed the
        # raw messages array. The orchestrator picks gmail_list if Gmail
        # is connected, then cascades to ms_mail_list, then Outlook +
        # read_screen if both are empty.
        # Phone-context pre-guard: the alternation below matches 'messages?'
        # as an email synonym, which would otherwise hijack "any messages
        # on my phone" / "read my texts" from phone_link_read_recent
        # further down. Bail early if the utterance has phone/SMS context.
        _has_phone_ctx = bool(_PHONE_CONTEXT_RE.search(t))
        if not _has_phone_ctx and re.search(
            r"\b("
            r"(?:summarize|summary\s+of|read|check|show|list|give\s+me|"
            r"tell\s+me\s+about|what(?:[’'´]?s|s'?s| is|\s+are)|"
            r"any|got\s+any|do\s+i\s+have)\s+"
            r"(?:my\s+|the\s+|new\s+|recent\s+)?(?:unread\s+)?"
            r"(?:emails?|messages?|mail|inbox|in[-\s]?box)"
            r"|"
            r"(?:summarize|read|check|catch\s+me\s+up\s+on|show\s+me)\s+"
            r"(?:my\s+)?(?:unread|inbox|mail|emails?)"
            r"|"
            r"(?:do\s+i\s+have|got|any)\s+(?:any\s+)?"
            r"(?:new\s+|recent\s+)?unread\b"
            r"|"
            r"what(?:[’'´]?s|s'?s| is)\s+(?:in\s+)?my\s+(?:inbox|mail)"
            r"|"
            # Provider-named requests: 'summarize my gmail unread',
            # 'check my work', 'read my personal inbox', 'show me my
            # yahoo'. Anchored by a verb + 'my' + provider keyword.
            r"(?:summarize|read|check|catch\s+me\s+up\s+on|show\s+me|"
            r"give\s+me|list|any|got\s+any|do\s+i\s+have)\s+"
            r"(?:my\s+|the\s+)?"
            r"(?:gmail|outlook|exchange|yahoo|icloud|hotmail|live|aol|"
            r"proton|fastmail|work|school|personal|main|primary)"
            r"(?:\s+(?:unread|emails?|inbox|mail|messages?|account))?"
            r")\b",
            t, flags=re.IGNORECASE,
        ):
            email_args: Dict[str, Any] = {
                "unread_only": True, "max": 50, "include_body": False,
            }
            # Optional account hint — extract from common phrasings
            # like "my gmail unread", "in my work account", "from my
            # personal email". Routes to Outlook COM's account filter.
            account = _extract_email_account(t)
            desc = "summarize unread emails (connector cascade)"
            if account:
                email_args["account"] = account
                desc = f"summarize unread in '{account}' (Outlook account)"
            return Step(
                tool="email_summary",
                args=email_args,
                layer="touchless",
                description=desc,
            )

        # ---- phone / SMS via Phone Link ("text X saying Y", "send a
        # text to X saying Y", "read my texts") --------------------------
        # Routes to phone_link_send_text / phone_link_read_recent
        # connector tools. Recipient resolution from memory happens at
        # the connector level (just like ms_mail_send).
        m = re.search(
            r"\b(?:send\s+(?:a\s+|me\s+)?)?(?:text|sms|imessage|"
            r"message\s+(?:via|on)\s+(?:phone|imessage|text))"
            r"\s+(?:to\s+)?"
            r"(?P<to>\+?[\d\s\-()]{7,}|[A-Za-z][A-Za-z0-9._\-]*)"
            r"\s+(?:saying|with\s+(?:the\s+)?message|that\s+says)\s+"
            r"(?P<body>.+)$",
            t, flags=re.IGNORECASE,
        )
        if m:
            recipient = m.group("to").strip(",.;")
            _PHONE_STOP = {"me", "him", "her", "them", "us", "the", "a",
                            "an", "my"}
            if recipient.lower() not in _PHONE_STOP:
                return Step(
                    tool="phone_link_send_text",
                    args={"to": recipient,
                          "body": m.group("body").strip()},
                    layer="connector",
                    description=f"send text via Phone Link to {recipient}",
                )
        # "read my (recent) texts" / "show me my latest texts" /
        # "any new messages on my phone" / "summarize my imessages"
        m = re.search(
            r"\b(?:read|show\s+me|check|summarize|any|got\s+any|"
            r"what\s+(?:are|'s|s))"
            r"\s+(?:any\s+)?(?:my\s+)?"
            r"(?:new\s+|recent\s+|latest\s+|unread\s+)?"
            r"(?:texts?|sms|imessages?"
            r"|messages\s+on\s+(?:my|the|this|that)\s+phone)"
            r"\b",
            lower,
        )
        if m:
            return Step(
                tool="phone_link_read_recent",
                args={"limit": 10},
                layer="connector",
                description="read recent texts via Phone Link",
            )

        # ---- self-setup ("set up KiCAD", "install kicad cli", "connect
        # spotify", "configure notion") --------------------------------
        # Generic: catches any '(set up|install|configure|connect)
        # <connector-name>' and routes to iris_setup_tool which finds the
        # connector by id/alias and calls its setup_self() — auto-discovers
        # binaries on PATH, walks OAuth flows, etc.
        # Path captured greedy-to-end so "set up X at C:\Program Files\..."
        # (with spaces) works. Use ORIGINAL `t` to preserve the path's
        # capitalization, not `lower`.
        m = re.search(
            r"\b(?:set\s*up|setup|install|configure|connect|enable)\b"
            r"\s+(?:the\s+|my\s+)?(?P<name>[a-z][a-z0-9_\-]*)"
            r"(?:\s+(?:cli|connector|api|integration))?"
            r"(?:\s+at\s+(?P<path>.+))?",
            lower,
        )
        if m:
            name = m.group("name")
            # Pronouns + common non-connector targets ('the meeting', 'the
            # project', etc.) — those clearly aren't connector ids.
            _SETUP_STOP = {
                "my", "the", "a", "an", "it", "that", "this",
                "all", "everything",
                "meeting", "appointment", "event", "project", "task",
                "file", "document", "doc", "sheet", "spreadsheet",
                "slide", "presentation", "session", "call", "alarm",
                "reminder", "notification", "schedule", "appointment",
            }
            if name not in _SETUP_STOP:
                args: Dict[str, Any] = {"name": name}
                # Pull the path from the ORIGINAL text (not lowercased) so
                # case-sensitive paths like 'C:\Program Files\...' survive.
                if m.group("path"):
                    orig_match = re.search(
                        r"\bat\s+(?P<orig_path>.+)$",
                        t, flags=re.IGNORECASE,
                    )
                    if orig_match:
                        args["path"] = orig_match.group("orig_path").strip()
                return Step(
                    tool="iris_setup_tool",
                    args=args,
                    layer="touchless",
                    description=f"set up {name}",
                )

        # ---- open the last-created artifact ("open it", "can you open
        # it", "show me that", "open the doc you just made") --------------
        # Resolves to whatever the planner most recently created (doc /
        # sheet / slide / OneNote / email draft, anything with a link).
        # Without this, 'open it' falls to realtime which has terrible
        # pronoun resolution and confabulates random plans.
        m = re.search(
            r"^(?:can\s+you\s+|could\s+you\s+|please\s+)*"
            r"(?:open|show(?:\s+me)?|pull\s+up|bring\s+up|view)\s+"
            r"(?:it|that|this|the\s+(?P<kind>doc(?:ument)?|sheet|spreadsheet|"
            r"slide(?:show|s)?|presentation|deck|onenote|page|note|file|"
            r"link|thing)(?:\s+(?:you|i|we)\s+(?:just\s+)?"
            r"(?:made|created|opened))?)"
            r"(?:\s+please)?\??\s*$",
            lower,
        )
        if m:
            raw_kind = (m.groupdict().get("kind") or "").strip()
            _KIND_ALIASES = {
                "document": "doc",
                "doc": "doc",
                "spreadsheet": "sheet",
                "sheet": "sheet",
                "slide": "slideshow",
                "slides": "slideshow",
                "slideshow": "slideshow",
                "presentation": "slideshow",
                "deck": "slideshow",
                "onenote": "OneNote page",
                "page": "OneNote page",
                "note": "OneNote page",
                "file": "file",
            }
            kind_hint = _KIND_ALIASES.get(raw_kind)
            open_args: Dict[str, Any] = (
                {"kind_hint": kind_hint} if kind_hint else {}
            )
            return Step(
                tool="iris_open_last",
                args=open_args,
                layer="touchless",
                description="open the most recently created artifact",
            )

        # ---- project remove ("remove demo project", "remove the project
        # called X", "delete the X project", "hide project Y", "forget the
        # X project") -------------------------------------------------
        # Route DIRECTLY to iris_remove_project so loose phrasings don't
        # have to round-trip through the LLM (which previously didn't see
        # iris_remove_project in its catalog and dropped these intents).
        # MUST sit ABOVE the contact-forget block — its
        # `(forget|delete|remove|wipe) NAME` pattern would otherwise steal
        # 'remove demo project' (matching NAME='demo').
        #
        # Patterns recognized (label is captured greedily to end, then
        # cleaned: strip trailing 'project', 'from cortex/your cortex/
        # your brain/your world', trailing punctuation):
        #   "remove demo project"           → label='demo'
        #   "remove project demo project"   → label='demo'
        #   "remove the project demo"       → label='demo'
        #   "remove the project called X"   → label='X'
        #   "delete the X project from your cortex" → label='X'
        #   "hide project X"                → label='X'
        #   "forget the project X"          → label='X'
        _PROJECT_REMOVE_VERB = r"(?:remove|delete|hide|forget|get\s+rid\s+of)"
        _PROJECT_TAIL_STRIP = re.compile(
            r"\s*(?:from\s+(?:my\s+|your\s+)?(?:cortex|brain|world|memory|"
            r"the\s+cortex|the\s+world))?\s*$",
            re.IGNORECASE,
        )
        project_patterns = (
            # "remove the project called/named/titled X"
            rf"^{_PROJECT_REMOVE_VERB}\s+(?:the\s+|my\s+)?project\s+"
            r"(?:called|named|titled)\s+(?P<label>.+)$",
            # "remove project X" / "remove the project X" / "remove project: X"
            rf"^{_PROJECT_REMOVE_VERB}\s+(?:the\s+|my\s+)?project[:\s]+"
            r"(?P<label>.+)$",
            # "remove the X project" / "delete X project"
            rf"^{_PROJECT_REMOVE_VERB}\s+(?:the\s+|my\s+)?"
            r"(?P<label>.+?)\s+project\b.*$",
            # "remove X from (my|your) cortex/brain/world"
            rf"^{_PROJECT_REMOVE_VERB}\s+(?:the\s+)?(?P<label>.+?)\s+"
            r"from\s+(?:my\s+|your\s+)?(?:cortex|brain|world)\b.*$",
        )
        for pat in project_patterns:
            m = re.search(pat, lower)
            if not m:
                continue
            label = m.group("label").strip()
            # Strip trailing "project" / cortex tail / punctuation.
            label = _PROJECT_TAIL_STRIP.sub("", label)
            label = re.sub(r"\s+project\s*$", "", label, flags=re.IGNORECASE)
            label = label.strip().rstrip(".!?,;:").strip()
            # Reject pronouns / non-labels.
            _PROJ_STOP = {"", "it", "that", "this", "my", "the", "a", "an",
                          "everything", "all", "them", "those", "these"}
            if label.lower() in _PROJ_STOP:
                continue
            # Reject single-letter labels (likely a misparse).
            if len(label) < 2:
                continue
            return Step(
                tool="iris_remove_project",
                args={"project_label": label},
                layer="iris",
                description=f"remove project: {label}",
            )

        # ---- contact forget ("forget Dani", "forget Dani's email",
        # "delete Vesko from memory", "you can forget about Mariya") ----
        # Lets the user purge memory facts without the CLI.
        #
        # Pre-guard: skip when the utterance is spreadsheet/cell context
        # ("remove Email from cell A2"). The verb+NAME regex otherwise
        # captures "Email" as a contact-to-forget and nukes real memory.
        if not _SHEET_CELL_CTX_RE.search(t):
            m = re.search(
                r"\b(?:forget|delete|remove|wipe)\b"
                r"(?:\s+about|\s+everything\s+about)?\s+"
                r"(?P<name>[A-Za-z][A-Za-z0-9._\-]*)"
                r"(?:'s|s)?\s*(?:email|address|info|details|number|"
                r"from\s+(?:memory|your\s+memory)|address\s+book)?",
                t, flags=re.IGNORECASE,
            )
        else:
            m = None
        if m:
            name = m.group("name")
            _STOP = {"my", "me", "your", "the", "a", "an", "his", "her",
                     "their", "our", "all", "everything", "it", "that"}
            # Also reject common nouns that show up as column headers or
            # cell values ("remove Email from cell A2", "delete Phone
            # from row 3"), and bare artifact head-nouns spoken alone
            # ("delete task", "forget note"). These are edits / clarify-
            # requests, not memory ops.
            _COMMON_NOUNS = {"email", "phone", "cell", "value", "row",
                             "column", "sheet", "range",
                             "task", "note", "doc", "document",
                             "spreadsheet", "slide", "slideshow",
                             "deck", "presentation", "event",
                             "meeting", "reminder", "item",
                             "form", "poll", "survey", "quiz",
                             "page", "file",
                             # Anchor words that appear right after
                             # delete/remove/forget when the target is
                             # an artifact: 'delete called drill-task'
                             # captures name='called' — never a person.
                             "called", "titled", "named", "about"}
            # Coined artifact identifiers ('drill-task', 'q4_note',
            # 'onboarding-doc') aren't real people — bail and let the
            # LLM planner resolve them against the session artifact
            # tracker (Orchestrator.find_artifact_by_name).
            _looks_like_artifact_name = (
                "-" in name or "_" in name
                or bool(_ARTIFACT_SUFFIX_RE.search(name))
            )
            if (name.lower() not in _STOP
                    and name.lower() not in _COMMON_NOUNS
                    and not _looks_like_artifact_name):
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

        # ---- Google identity (whoami + birthday) --------------------------
        # MUST sit ABOVE iris_lookup_contact so 'what is my email' /
        # 'what's my google name' don't get stolen by the contact
        # 'X's email' regex. Both routes are deterministic — no LLM call.
        if re.search(
            r"^\s*(?:please\s+|hey\s+iris\s+|iris\s+)?"
            r"(?:who\s+am\s+i"
            r"(?:\s+(?:signed|logged|currently\s+signed|currently\s+logged)\s+"
            r"(?:in|on)(?:\s+as)?)?"
            r"|what(?:[’'´]?s|s| is)\s+my\s+"
            r"(?:google\s+|primary\s+)?"
            r"(?:email(?:\s+address)?|account|name|profile|"
            r"user(?:name)?\s+on\s+google)"
            r"|what\s+account\s+(?:am\s+i\s+(?:on|using)|"
            r"is\s+(?:this|signed\s+in)))"
            r"\s*\??\s*$",
            lower,
        ):
            return Step(
                tool="google_whoami",
                args={},
                layer="connector",
                description="who am I signed in as (Google)")
        # Birthday self-lookup. 'when's my birthday' / 'what's my dob' /
        # 'how old am i'. Strictly second-person possessive ('my').
        if re.search(
            r"^\s*(?:please\s+|hey\s+iris\s+|iris\s+)?"
            r"(?:when(?:[’'´]?s|s| is)\s+my\s+(?:birthday|bday|b[-\s]?day)"
            r"|what(?:[’'´]?s|s| is)\s+my\s+"
            r"(?:birthday|bday|date\s+of\s+birth|dob)"
            r"|how\s+old\s+am\s+i"
            r"|when\s+do\s+i\s+turn\s+\d+)"
            r"\s*\??\s*$",
            lower,
        ):
            return Step(
                tool="google_my_birthday",
                args={},
                layer="connector",
                description="my Google profile birthday")

        # ---- YouTube readonly (playlists / watch later / subscriptions) ---
        # All three routes require explicit YouTube context ('youtube' /
        # 'yt') to avoid colliding with Spotify ('my playlists') and
        # other broad phrasings.
        if (re.search(
                r"\b(?:list|show(?:\s+me)?|tell\s+me|give\s+me|"
                r"what(?:[’'´]?s|\s+is|\s+are)|read|check)\s+"
                r"(?:me\s+)?(?:all\s+)?(?:my|the)\s+"
                r"(?:youtube\s+|yt\s+)playlists?\b", lower)
            and "spotify" not in lower):
            return Step(
                tool="youtube_my_playlists",
                args={},
                layer="connector",
                description="list my YouTube playlists")
        # Watch Later — emits Step with playlist_id='WL'. Connector
        # returns a friendly fallback URL when WL is API-inaccessible.
        if re.search(
            r"\b(?:what(?:[’'´]?s|s| is)\s+in|show(?:\s+me)?|read|check|"
            r"open|view)\s+(?:me\s+)?(?:my\s+)?watch[\s-]?later\b",
            lower,
        ):
            return Step(
                tool="youtube_playlist_items",
                args={"playlist_id": "WL"},
                layer="connector",
                description="show YouTube Watch Later")
        # Subscriptions — anchored on 'subscribed' or 'subscriptions' +
        # 'youtube' / 'channel(s)' context.
        if (re.search(
                r"\b(?:who\s+am\s+i\s+subscribed\s+to"
                r"|(?:list|show|what\s+are)\s+(?:me\s+)?my\s+"
                r"(?:youtube\s+)?subscriptions?"
                r"|what\s+channels\s+do\s+i\s+(?:follow|subscribe\s+to))\b",
                lower)
            and ("youtube" in lower or "yt" in lower or "channel" in lower
                 or "subscrib" in lower)):
            return Step(
                tool="youtube_subscriptions",
                args={},
                layer="connector",
                description="list my YouTube subscriptions")

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

        # ---- Google Contacts (People API) -------------------------------
        # Sits BELOW iris_lookup_contact so memory lookups stay fast and
        # the People API only fires when the user explicitly names
        # 'contacts' / 'address book' OR asks a generic 'who is X' OR
        # asks for a phone number.
        _CONTACT_NAME_STOP = {"my", "me", "your", "the", "a", "an", "his",
                              "her", "their", "our", "anyone", "someone",
                              "nobody", "him", "her"}

        # ---- contacts_list with prefix filter ('list contacts starting
        # with A', 'my A contacts', 'anyone whose name starts with C').
        # Sits ABOVE the plain contacts_list matcher so the prefix arg is
        # captured before the bare list-verb match swallows the phrase.
        # Three shapes:
        #   (i)  verb + "contacts" + "start(ing|s|beginning|begins) with X"
        #   (ii) "<Letter> contacts" possessive shorthand
        #   (iii) "contacts whose (first|last) name starts/begins with X"
        # All normalize the captured prefix to uppercase for a clean spoken
        # summary; the connector lowercases for the actual comparison.
        _prefix_letter: Optional[str] = None
        # (i-a) Verb-anchored: 'list contacts starting with A',
        # 'show me contacts that begin with B', 'how many contacts
        # start with C'.
        m = re.search(
            r"\b(?:list|show|give\s+me|find|who\s+are|which\s+of|"
            r"any\s+of\s+my|anyone|how\s+many|read|tell\s+me)\b"
            r".*?\bcontacts?\b.*?"
            r"\b(?:start(?:s|ing)?|begin(?:s|ning)?)\s+with\s+"
            r"[\"']?(?P<prefix>[A-Za-z])[\"']?\b",
            lower,
        )
        # (i-b) Noun-first: 'contacts that start with B', 'contacts
        # starting with A', 'contacts beginning with C'. No verb —
        # some users omit the imperative. Requires 'contacts' explicit.
        if m is None:
            m = re.search(
                r"\bcontacts?\s+(?:that\s+|which\s+|who\s+)?"
                r"(?:start(?:s|ing)?|begin(?:s|ning)?)\s+with\s+"
                r"[\"']?(?P<prefix>[A-Za-z])[\"']?\b",
                lower,
            )
        if m is None:
            # (ii-a) 'contacts whose (first|last|given|family|) name
            # starts/begins with X' — possessive with 'contacts' anchor.
            m = re.search(
                r"\bcontacts?\s+whose\s+"
                r"(?:name|first\s+name|last\s+name|given\s+name|"
                r"family\s+name)\s+"
                r"(?:start(?:s|ing)?|begin(?:s|ning)?)\s+with\s+"
                r"[\"']?(?P<prefix>[A-Za-z])[\"']?",
                lower,
            )
        if m is None:
            # (ii-b) 'anyone / anybody / someone / who / who's whose
            # (first|last)? name starts with X'. No 'contacts' word
            # needed — the intent is unambiguously an address-book
            # lookup at this point in the classifier (we already
            # filtered out mail/task/reminder/etc. above).
            m = re.search(
                r"\b(?:anyone|anybody|someone|who|who[’'´]?s)\s+"
                r"whose\s+"
                r"(?:name|first\s+name|last\s+name|given\s+name|"
                r"family\s+name)\s+"
                r"(?:start(?:s|ing)?|begin(?:s|ning)?)\s+with\s+"
                r"[\"']?(?P<prefix>[A-Za-z])[\"']?",
                lower,
            )
        if m is None:
            # Possessive shorthand: 'list all A contacts' / 'my A
            # contacts' / 'the A contacts'. Require an ARTICLE / verb
            # anchor before "<Letter> contacts" so a bare capital in a
            # sentence like 'my Boston contacts' doesn't get grabbed
            # (Boston is multi-char; we already require single letter,
            # but a stray "I contacts" would still be nonsense). We use
            # `t` (original case) here so a lowercase 'a' -> 'A' is
            # only captured when it's really a spoken letter, not the
            # indefinite article; therefore require the letter to be
            # UPPER-case in the original text OR the phrase to include
            # 'letter' word.
            m = re.search(
                r"\b(?:list|show|give\s+me|find|any|all)\s+"
                r"(?:of\s+)?(?:my\s+|the\s+)?"
                r"(?P<prefix>[A-Z])\s+contacts?\b",
                t,
            )
            if m is None:
                m = re.search(
                    r"\b(?:my|the)\s+(?P<prefix>[A-Z])\s+contacts?\b",
                    t,
                )
            if m is None:
                # Explicit 'letter' anchor: 'contacts starting with the
                # letter a' / 'my letter A contacts' — accept
                # lower-case letters here since the word 'letter'
                # disambiguates.
                m = re.search(
                    r"\bcontacts?\s+"
                    r"(?:start(?:s|ing)?|begin(?:s|ning)?)\s+with\s+"
                    r"(?:the\s+)?letter\s+"
                    r"[\"']?(?P<prefix>[A-Za-z])[\"']?\b",
                    lower,
                )
                if m is None:
                    m = re.search(
                        r"\b(?:the\s+)?letter\s+"
                        r"(?P<prefix>[A-Za-z])\s+contacts?\b",
                        lower,
                    )
        if m:
            _prefix_letter = m.group("prefix").upper()
            return Step(
                tool="contacts_list",
                args={"starts_with": _prefix_letter},
                layer="connector",
                description=(
                    f"list contacts starting with {_prefix_letter}"))

        # ---- contacts_list: 'list my contacts', 'show me my contacts',
        # 'who's in my contacts', 'how many contacts do I have'.
        # Anchored to bare 'contacts' with a list-verb — must NOT match
        # 'search my contacts for X' (that's contacts_search) so the
        # verb alternation is restricted to list/show/read/give-me and
        # there's no name capture. ------------------------------------
        if re.search(
                r"\b(?:list|show(?:\s+me)?|read|give\s+me|"
                r"how\s+many|what[’'´]?s\s+in|who[’'´]?s\s+in)\s+"
                r"(?:all\s+)?(?:my\s+|the\s+)?"
                r"(?:contacts?|address\s+book)\b"
                r"(?:\s+do\s+i\s+have|\s+are\s+(?:there|in\s+there))?"
                r"(?:\s+(?:please|list))?"
                r"\s*[.!?]*$",
                lower):
            return Step(
                tool="contacts_list",
                args={},
                layer="connector",
                description="list Google Contacts")

        # Pre-guard: skip ALL contacts_search patterns (a/b/c) when the
        # utterance is really about a spreadsheet cell (e.g. "what is the
        # value of cell A2 in the sheet called testing"). Otherwise a
        # bare capitalised word ("Testing") or the cell token ("A2") gets
        # captured as a contact name and burns a People API call.
        _skip_contacts_search = bool(_SHEET_CELL_CTX_RE.search(t))

        # (a) Explicit contacts-search by name: 'find Dani in my contacts',
        # 'search my contacts for Dani', 'who is Dani in my contacts'.
        m = None
        if not _skip_contacts_search:
            m = re.search(
                r"\b(?:find|look\s*up|search|who\s+is)\b"
                r"[^?]*?\b(?P<name>[A-Z][A-Za-z\-]+(?:\s+[A-Z][A-Za-z\-]+)?)"
                r"[^?]*?\b(?:contacts?|address\s+book)\b",
                t,
            )
            if m is None:
                m = re.search(
                    r"\b(?:contacts?|address\s+book)\b"
                    r"[^?]*?\b(?:for\s+)?(?P<name>[A-Z][A-Za-z\-]+(?:\s+[A-Z][A-Za-z\-]+)?)\b",
                    t,
                )
        if m:
            name = m.group("name").strip()
            if name.lower() not in _CONTACT_NAME_STOP and len(name) >= 2:
                return Step(
                    tool="contacts_search",
                    args={"query": name},
                    layer="connector",
                    description=f"search Google Contacts for {name}")

        # (b) Generic 'who is X' / 'do I know X' / 'tell me about X' — no
        # explicit info-type. Wider net than the memory-lookup pattern.
        m = None
        if not _skip_contacts_search:
            m = re.search(
                r"^(?:who(?:[’'´]?s|\s+is)|do\s+i\s+know|tell\s+me\s+about)\s+"
                r"(?P<name>[A-Z][A-Za-z\-]+(?:\s+[A-Z][A-Za-z\-]+)?|"
                r"\"[^\"]+\"|'[^']+')"
                r"\s*\??\s*$",
                t,
            )
        if m:
            name = m.group("name").strip().strip("'\"")
            if name.lower() not in _CONTACT_NAME_STOP and len(name) >= 2:
                return Step(
                    tool="contacts_search",
                    args={"query": name},
                    layer="connector",
                    description=f"who is {name} (Google Contacts)")

        # (c) Phone-number lookup — mirrors the email-lookup regex but with
        # phone/cell/mobile/number in place of email/address.
        m = None
        if not _skip_contacts_search:
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
                r"(?:[’'´]?s|s)?\s+"
                r"(?:phone|phone\s+number|number|cell|cell\s+phone|mobile|"
                r"mobile\s+number)\b",
                t, flags=re.IGNORECASE,
            )
        if m:
            name = m.group("name")
            if name.lower() not in _CONTACT_NAME_STOP:
                return Step(
                    tool="contacts_search",
                    args={"query": name},
                    layer="connector",
                    description=f"look up {name}'s phone (Google Contacts)")

        # (d) Contact creation — 'add John to my contacts with email
        # john@x.com', 'save Mariya as a contact with phone 555-1234'.
        # Require AT LEAST one of email/phone.
        _EMAIL_RE = r"(?P<email>[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,})"
        _PHONE_RE = r"(?P<phone>\+?[\d][\d\s\-()]{6,})"
        contact_create_patterns = (
            # 'add John to my contacts (with email X) (and phone Y)'
            rf"^(?:add|put|save)\s+"
            rf"(?P<name>[A-Z][A-Za-z\-]+(?:\s+[A-Z][A-Za-z\-]+)?)\s+"
            rf"to\s+my\s+contacts"
            rf"(?:\s+with\s+email\s+{_EMAIL_RE})?"
            rf"(?:\s+(?:and\s+)?(?:with\s+)?phone\s+{_PHONE_RE})?"
            rf"\s*[.!?]*$",
            # 'add/create/save (a [new]) contact (for|named|called) NAME
            # (with) (email X) (phone Y)' — email/phone interchangeable
            rf"^(?:add|create|save|make)\s+(?:a\s+(?:new\s+)?)?contact\s+"
            rf"(?:for\s+|named\s+|called\s+)?"
            rf"(?P<name>[A-Z][A-Za-z\-]+(?:\s+[A-Z][A-Za-z\-]+)?)"
            rf"(?:\s+with)?"
            rf"(?:\s+email\s+{_EMAIL_RE})?"
            rf"(?:\s+(?:and\s+|with\s+)?phone\s+{_PHONE_RE})?"
            rf"\s*[.!?]*$",
            # 'add/create/save (a [new]) contact (for|named|called) NAME
            # phone Y email X' — phone first
            rf"^(?:add|create|save|make)\s+(?:a\s+(?:new\s+)?)?contact\s+"
            rf"(?:for\s+|named\s+|called\s+)?"
            rf"(?P<name>[A-Z][A-Za-z\-]+(?:\s+[A-Z][A-Za-z\-]+)?)"
            rf"(?:\s+with)?"
            rf"(?:\s+phone\s+{_PHONE_RE})?"
            rf"(?:\s+(?:and\s+|with\s+)?email\s+{_EMAIL_RE})?"
            rf"\s*[.!?]*$",
        )
        for pat in contact_create_patterns:
            m = re.search(pat, t)
            if not m:
                continue
            groups = m.groupdict()
            email = (groups.get("email") or "").strip()
            phone = (groups.get("phone") or "").strip()
            if not email and not phone:
                # At-least-one rule.
                continue
            name = m.group("name").strip()
            if name.lower() in _CONTACT_NAME_STOP:
                continue
            name_parts = name.split()
            given = name_parts[0]
            family = " ".join(name_parts[1:]) if len(name_parts) > 1 else ""
            args: Dict[str, Any] = {"given_name": given}
            if family:
                args["family_name"] = family
            if email:
                args["email"] = email
            if phone:
                args["phone"] = phone
            return Step(
                tool="contacts_create",
                args=args,
                layer="connector",
                description=f"create Google Contact: {name}")

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

    def classify_chain(self, text: str) -> Optional[List[Step]]:
        """Like classify(), but returns a list of Steps so a single utterance
        like 'make a doc titled X with body Y and open it' produces both the
        gdocs_create AND an iris_open_last to launch the just-created link.

        Returns:
          - None when the text doesn't match any classifier pattern;
          - [single_step] when matched but no chain suffix detected (caller
            treats this exactly like classify() returning that Step);
          - [create_step, open_step] when a trailing open suffix was stripped
            and the matched tool is an artifact-creating tool whose link can
            be opened next.
        """
        t = (text or "").strip()
        if not t:
            return None
        clean, had_open = strip_open_suffix(t)
        base = self.classify(clean if had_open else t)
        if base is None:
            return None
        if not had_open:
            return [base]
        kind = _CHAIN_OPEN_KIND_BY_TOOL.get(base.tool)
        if not kind:
            return [base]
        open_step = Step(
            tool="iris_open_last",
            args={"kind_hint": kind},
            layer="touchless",
            id=2,
            depends_on=[1],
            description=f"open just-created {kind}",
        )
        base.id = 1
        return [base, open_step]
