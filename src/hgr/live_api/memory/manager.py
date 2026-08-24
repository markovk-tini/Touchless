"""MemoryManager — the single entry point for the rest of Iris.

Two operations:

  record(user_text, plan, steps, results, message)
      Async-ish: extracts facts synchronously (fast — pure regex/dict work)
      and writes them, then spins a background thread for the episodic
      embed + insert so the user-visible reply isn't blocked by the
      embedder's HTTP round-trip.

  recall(goal_text, k=3) -> {"episodes": [...], "facts": [...], "context": str}
      Returns the top-k episodic matches (cosine sim over embeddings) and
      any semantic facts whose key appears in the goal text. The
      "context" string is a pre-rendered block ready to drop into the
      planner / synthesizer prompt.

Storage path defaults to %LOCALAPPDATA%\\Touchless\\memory.db on Windows;
on other platforms falls back to ~/.touchless/memory.db. Override via
TOUCHLESS_MEMORY_DB.

Author: Konstantin Markov
"""
from __future__ import annotations

import json
import os
import re
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from .. import cortex_emit
from .embedder import Embedder, cosine_sim, default_embedder
from .extractor import extract_facts
from .llm_extractor import (
    confidence_threshold as _llm_confidence_threshold,
    enabled as _llm_extraction_enabled,
    extract_facts_from_conversation,
)
from .project_memory import ProjectChunk, ProjectMemoryStore
from .store import EpisodicRow, MemoryStore, SemanticRow


def default_memory_path() -> Path:
    override = os.environ.get("TOUCHLESS_MEMORY_DB")
    if override:
        return Path(override)
    if os.name == "nt":
        base = os.environ.get("LOCALAPPDATA") or str(Path.home())
        return Path(base) / "Touchless" / "memory.db"
    return Path.home() / ".touchless" / "memory.db"


# How many recent episodes to scan when ranking by cosine similarity.
_RECALL_SCAN = 200
# Minimum cosine similarity to surface an episode as relevant.
_RECALL_MIN_SIM = 0.1
# Max chars of context block injected into a prompt.
_CONTEXT_MAX_CHARS = 600

# Project-RAG: how many chunks per project to surface in recall(), and
# how many chars of project context we'll allow in the rendered block.
_RAG_TOP_K = 4
_RAG_CONTEXT_MAX_CHARS = 1200
# When set, recall() does a broad cross-project search if no project
# label matched. Off by default — see project_memory.py for why.
_RAG_BROAD_ENV = "TOUCHLESS_RAG_BROAD"

# Pattern detection (OPEN_ISSUES #10): every N successful observations
# we run find_repeated_patterns and surface any newly-seen patterns to
# the cortex bridge. Default 5 keeps it cheap; the bulk of the work is
# the embedder call inside find_repeated_patterns, which is itself
# cached/cheap for repeated queries.
_PATTERN_CHECK_INTERVAL = 5
# Cap how many patterns we surface in one batch so a sudden surge
# can't spam the simulator with dozens of signals at once.
_PATTERN_EMIT_MAX = 5

# Episodic outcome cap. SQLite's `outcome TEXT` column has no length
# constraint — this number is purely policy to keep the rendered
# context bounded. Bumped from 600 -> 8000 so a chat-generated email
# / poem / code snippet survives without being chopped mid-body.
_OUTCOME_MAX_CHARS = 8000
# Artifact semantic-row value cap. Stays scannable in the recall path
# while comfortably fitting a typical email/poem draft body.
_ARTIFACT_VALUE_MAX_CHARS = 4000

# Heuristic gate for record_chat_turn: only persist the assistant's
# full reply as an episodic + artifact when the USER asked the model
# to GENERATE durable content. Keeps plain chitchat ("what's the
# weather", "thanks") out of the episodic store.
_DRAFT_VERB_RE = re.compile(
    r"\b("
    r"draft(?:ing|s|ed)?|"
    r"compos(?:e|ing|es|ed)|"
    r"writ(?:e|ing|es)|wrote|"
    r"author(?:ing|s|ed)?|"
    r"generat(?:e|ing|es|ed)|"
    r"make me (?:an?|the|some)|"
    r"give me (?:an?|the|some)|"
    r"creat(?:e|ing|es|ed) (?:an?|the|some)|"
    r"summari[sz](?:e|ing|es|ed)|"
    r"rewrit(?:e|ing|es)|rewrote|"
    r"paraphras(?:e|ing|es|ed)|"
    r"translat(?:e|ing|es|ed)|"
    r"outlin(?:e|ing|es|ed)"
    r")\b",
    re.IGNORECASE,
)
_DRAFT_NOUN_RE = re.compile(
    r"\b("
    r"emails?|messages?|drafts?|"
    r"poems?|haikus?|songs?|lyrics?|"
    r"essays?|articles?|posts?|tweets?|blogs?|captions?|"
    r"letters?|notes?|memos?|memo|repl(?:y|ies)|responses?|"
    r"speech|speeches|toasts?|eulog(?:y|ies)|"
    r"summar(?:y|ies)|outlines?|"
    r"stor(?:y|ies)|scripts?|"
    r"code|snippets?|functions?|classes?|"
    r"sql|quer(?:y|ies)|regex|"
    r"agendas?|itinerar(?:y|ies)|"
    r"recipes?|paragraphs?|sentences?"
    r")\b",
    re.IGNORECASE,
)
# Map noun -> short canonical kind slug used for `artifact <kind>:<topic>`
# rows. Anything not matched falls back to "text".
_DRAFT_KIND_MAP = (
    ("email", ("email", "emails", "reply", "replies", "response", "responses")),
    ("message", ("message", "messages")),
    ("poem", ("poem", "poems", "haiku", "haikus", "song", "songs",
              "lyric", "lyrics")),
    ("letter", ("letter", "letters")),
    ("note", ("note", "notes", "memo", "memos")),
    ("essay", ("essay", "essays", "article", "articles", "post", "posts",
               "blog", "blogs")),
    ("tweet", ("tweet", "tweets", "caption", "captions")),
    ("speech", ("speech", "speeches", "toast", "toasts",
                "eulogy", "eulogies")),
    ("summary", ("summary", "summaries", "outline", "outlines")),
    ("story", ("story", "stories", "script", "scripts")),
    ("code", ("code", "snippet", "snippets", "function", "functions",
              "class", "classes", "sql", "query", "queries", "regex")),
    ("agenda", ("agenda", "agendas", "itinerary", "itineraries")),
    ("recipe", ("recipe", "recipes")),
    ("paragraph", ("paragraph", "paragraphs", "sentence", "sentences")),
    ("draft", ("draft", "drafts")),
)
# Topic-slug extraction: strip these so 'about the leak under the
# sink' boils down to 'leak'. Tuned for the artifact recall use-case,
# NOT for general NLP — false positives just produce slightly longer
# slugs, which is fine.
_TOPIC_STOPWORDS = frozenset({
    "a", "an", "the", "my", "our", "your", "their", "his", "her", "its",
    "this", "that", "these", "those", "some", "any", "all",
    "to", "for", "of", "in", "on", "at", "by", "with", "about",
    "regarding", "concerning", "saying", "telling", "asking",
    "please", "kindly", "really", "very", "just",
    "and", "or", "but", "so",
    "today", "tomorrow", "tonight", "yesterday",
})


def _looks_like_draft_request(user_text: str) -> bool:
    """True when the user's message is asking the model to GENERATE
    durable text content (email, poem, message, code, summary, ...).
    Used to gate the chat-episodic write path so we don't blow up
    the store with every chitchat turn."""
    if not user_text:
        return False
    t = user_text.strip()
    if len(t) < 6:
        return False
    return bool(_DRAFT_VERB_RE.search(t) and _DRAFT_NOUN_RE.search(t))


# Phrases that mark an assistant reply as a clarifying question rather
# than a delivered draft. When the same-turn assistant reply looks like
# a clarifying question, we SUPPRESS the artifact write but ARM the
# pending-draft latch so a later delivery turn can be captured.
_CLARIFY_HINT_RE = re.compile(
    r"\b("
    r"i need more|need more details?|need a bit more|"
    r"give me (?:a )?(?:quick )?(?:rundown|summary|details?)|"
    r"any specifics?|any details?|"
    r"what'?s going on|what'?s the (?:issue|problem|situation|context|gist)|"
    r"could you (?:tell|share|give)|can you (?:tell|share|give)|"
    r"a few details?|some details?|"
    r"who is it (?:to|for)|who'?s it (?:to|for)|"
    r"what would you like|what should i|"
    r"any (?:tone|style|length) preference"
    r")\b",
    re.IGNORECASE,
)
_CLARIFY_MAX_CHARS = 280


def _looks_like_clarifying_question(assistant_text: str) -> bool:
    """True when the assistant reply is a SHORT clarifying question
    rather than a delivered draft. Used to suppress same-turn artifact
    writes while still arming the pending-draft latch."""
    if not assistant_text:
        return False
    t = assistant_text.strip()
    if not t:
        return False
    if len(t) > _CLARIFY_MAX_CHARS:
        return False
    # Ends with a question mark, OR contains a known clarify phrase.
    if t.rstrip().endswith("?"):
        return True
    return bool(_CLARIFY_HINT_RE.search(t))


# Header markers that strongly indicate the assistant produced a
# delivered draft (Subject: / Dear ... / Hi [Name] / Hello ...).
_DRAFT_HEADER_RE = re.compile(
    r"(?mi)^\s*("
    r"subject\s*:|"
    r"dear\s+[A-Za-z\[]|"
    r"hi\s+\[|"
    r"hi\s+[A-Z][a-z]|"
    r"hello\s+[A-Za-z\[]|"
    r"to\s*:\s*\S|"
    r"from\s*:\s*\S"
    r")",
)
_DRAFT_MIN_CHARS = 300
_DRAFT_CODE_FENCE_RE = re.compile(r"```")


def _looks_like_delivered_draft(assistant_text: str,
                                 kind: Optional[str] = None) -> bool:
    """True when the assistant reply looks like a delivered draft body
    rather than chat / clarification. Heuristic:

      - >= _DRAFT_MIN_CHARS chars, AND
      - contains a header marker (Subject: / Dear / Hi [Name] / Hello /
        To: / From:), OR
      - contains a code fence (for code-kind drafts), OR
      - is multi-paragraph (>=2 blank-line separators).

    When ``kind == 'code'`` we accept any reply that contains a code
    fence even below the length threshold — code snippets are
    legitimately short."""
    if not assistant_text:
        return False
    t = assistant_text.strip()
    if not t:
        return False
    # Code drafts: a fenced block is enough regardless of length.
    if kind == "code" and _DRAFT_CODE_FENCE_RE.search(t):
        return True
    if len(t) < _DRAFT_MIN_CHARS:
        return False
    if _DRAFT_HEADER_RE.search(t):
        return True
    if _DRAFT_CODE_FENCE_RE.search(t):
        return True
    # Multi-paragraph body: 2+ blank-line separators.
    if len(re.findall(r"\n\s*\n", t)) >= 2:
        return True
    return False


def _classify_draft_kind(user_text: str) -> str:
    """Map the user's request to a short canonical kind slug
    (email / poem / code / summary / ...). The VERB wins over the
    noun when it's diagnostic ('summarize this' -> summary, even if
    'article' also appears), since the verb names the OUTPUT and the
    noun often names the INPUT. Falls back to noun-table scan, then
    to 'text'."""
    if not user_text:
        return "text"
    lt = user_text.lower()
    # Verb-first overrides: when the verb itself names the output kind,
    # trust it. (E.g. 'summarize this article' is a SUMMARY of an
    # article, not an article.) Keep this list tight to verbs whose
    # output kind is unambiguous.
    if re.search(r"\bsummari[sz](?:e|ing|es|ed)\b", lt):
        return "summary"
    if re.search(r"\boutlin(?:e|ing|es|ed)\b", lt):
        return "summary"
    if re.search(r"\btranslat(?:e|ing|es|ed)\b", lt):
        return "text"
    if re.search(r"\bparaphras(?:e|ing|es|ed)\b", lt):
        return "text"
    for slug, nouns in _DRAFT_KIND_MAP:
        for n in nouns:
            if re.search(rf"\b{re.escape(n)}\b", lt):
                return slug
    return "text"


def _topic_slug_from_request(user_text: str) -> str:
    """Pull the topic phrase out of the user's request — typically the
    object of 'about' / 'to' / 'for'. Returns a short kebab-cased
    slug suitable for use as a semantic-row key suffix. Empty string
    when nothing useful can be extracted (caller should fall back to
    a generic key like the kind alone)."""
    if not user_text:
        return ""
    t = user_text.strip()
    # Prefer 'about <phrase>' / 'regarding <phrase>' / 'on <phrase>'.
    m = re.search(
        r"\b(?:about|regarding|concerning|on)\s+([A-Za-z][\w'\- ]{1,80})",
        t, re.IGNORECASE,
    )
    phrase = m.group(1) if m else ""
    if not phrase:
        # Fall back to 'to <recipient> ...' — strip the recipient and
        # try to find a topic clause after it.
        m = re.search(
            r"\bto\s+[A-Za-z][\w'\- ]{0,40}?\s+"
            r"(?:about|regarding|concerning|saying|telling)\s+"
            r"([A-Za-z][\w'\- ]{1,80})",
            t, re.IGNORECASE,
        )
        if m:
            phrase = m.group(1)
    if not phrase:
        # Last resort: 'for <phrase>' (e.g. 'a poem for mom').
        m = re.search(r"\bfor\s+([A-Za-z][\w'\- ]{1,80})",
                      t, re.IGNORECASE)
        phrase = m.group(1) if m else ""
    if not phrase:
        return ""
    tokens: List[str] = []
    for raw in re.split(r"[^A-Za-z0-9'\-]+", phrase):
        w = raw.strip("-' ").lower()
        if not w or w in _TOPIC_STOPWORDS:
            continue
        tokens.append(w)
        if len(tokens) >= 6:
            break
    return "-".join(tokens)


class MemoryManager:
    """Wraps the store + embedder. Threadsafe; non-blocking writes."""

    def __init__(self, store: Optional[MemoryStore] = None,
                 embedder: Optional[Embedder] = None,
                 async_writes: bool = True,
                 logger: Any = None,
                 project_store: Optional[ProjectMemoryStore] = None) -> None:
        self._store = store or MemoryStore(default_memory_path())
        self._embedder = embedder or default_embedder()
        self._async = async_writes
        self._logger = logger
        # Optional — if None, recall() simply skips the project_context
        # branch. Wiring is opt-in so existing tests (and any caller
        # that doesn't want RAG) keep working unchanged.
        self._project_store = project_store
        # Pattern surfacing (OPEN_ISSUES #10): every Nth observed
        # conversation we run find_repeated_patterns and fire a
        # patternAdded signal on the active cortex bridge for any
        # pattern we haven't seen before in this process. Kept cheap by
        # rate-limiting via _PATTERN_CHECK_INTERVAL and by skipping
        # entirely when no bridge is active.
        self._pattern_check_count = 0
        self._seen_pattern_labels: set = set()
        # Draft-intent latch: when a user turn matches
        # _looks_like_draft_request but the SAME-turn assistant reply
        # is a clarifying question (not a delivered draft), we arm this
        # latch with the original intent (kind+topic+user_text) so the
        # NEXT assistant reply that actually contains a draft body can
        # be promoted to an artifact keyed by the original topic.
        # Cleared on successful delivery or after ttl_turns elapses.
        # See _observe_safe for the state machine.
        self._pending_draft: Optional[Dict[str, Any]] = None
        # Lock for the latch — _observe_safe runs on a worker thread.
        self._pending_draft_lock = threading.Lock()
        # One-off cleanup: purge the bogus "clarifying-question masquerading
        # as draft" artifact rows written by the pre-latch gate (see
        # _purge_bogus_clarify_artifacts). Idempotent and cheap.
        try:
            self._purge_bogus_clarify_artifacts()
        except Exception as exc:  # pragma: no cover - defensive
            if self._logger:
                self._logger.exception(
                    "memory_purge_clarify_artifacts_failed", exc)

    # ---- session-start context summary -----------------------------------
    # Provenance kinds that produce mostly noise (tool-usage stats, machine
    # config) and shouldn't crowd out durable user-revealed facts in the
    # session summary. Still queryable via Tier 2 recall; just not auto-
    # injected at session start.
    _SUMMARY_SKIP_SOURCE_KINDS = frozenset({
        "tool_pattern", "git_config",
    })
    # Fact kinds that rarely help recall ("here's a doc URL from 3 weeks
    # ago"). Excluded from the session summary; the planner can still pull
    # them via direct find_facts lookups.
    _SUMMARY_SKIP_KINDS = frozenset({"artifact"})
    # Key-substring fragments that mark ephemeral / system-generated rows
    # under the catch-all 'fact' kind (screenshot paths, doc URLs, etc.).
    # These crowd out durable user-revealed facts (pet info, office) under
    # the same kind.
    _SUMMARY_SKIP_FACT_KEY_FRAGMENTS = (
        "screenshot", "url", "operating-system",
        "google-doc-", "google-drive", "google-spreadsheet",
        "stated-", "-stored",
    )
    # Identity-shaped kinds get a reserved top-N slot in the session
    # summary so durable rows (place=office, relation=professor) can't be
    # crowded out by recent noise (duplicate person rows, schedule churn)
    # under a pure-recency cap. See summary_for_session.
    _SUMMARY_IDENTITY_KINDS = ("place", "relation", "role", "course", "alias")
    _SUMMARY_IDENTITY_RESERVE = 8
    _SUMMARY_PERSON_RESERVE = 6
    _SUMMARY_IDENTITY_VALUE_NOISE = (
        "spreadsheet", "google sheet", "google doc", "google docs",
        "google slide", "presentation named", "document named",
    )

    def summary_for_session(self, max_facts: int = 30,
                            max_chars: int = 1500) -> str:
        """Render a compact 'what Iris knows about this user' note for
        realtime session start. Lets realtime answer recall questions
        like 'where's my office?' directly, without needing to enter a
        Tier 2 planning round-trip.

        Returns empty string when memory is empty / nothing useful to say.
        Caps at 30 facts (recent first) and 1500 chars so the session
        prompt doesn't bloat."""
        try:
            # Over-fetch and filter so noisy tool_pattern / artifact rows
            # can't push out durable user-revealed facts (pet info, office
            # location, prof name) when memory grows. We still cap the
            # rendered output at max_facts after filtering.
            raw = self._store.find_facts(limit=max(max_facts * 6, 200))
        except Exception:
            return ""
        def _is_ephemeral_fact(row: Any) -> bool:
            if (row.kind or "") != "fact":
                return False
            klow = (row.key or "").lower()
            return any(frag in klow
                       for frag in self._SUMMARY_SKIP_FACT_KEY_FRAGMENTS)

        def _is_skipped_artifact(row: Any) -> bool:
            # Tool-link artifacts (Google Doc/Sheet/Slides URLs) are
            # noisy in the session summary, but LLM-draft artifacts
            # (provenance source_kind='llm_draft') are exactly what
            # we want to surface so 'show me the email I drafted'
            # is answerable from the summary too.
            if (row.kind or "") != "artifact":
                return False
            return (row.source_kind or "") != "llm_draft"

        filtered = [f for f in raw
                    if (f.source_kind or "") not in self._SUMMARY_SKIP_SOURCE_KINDS
                    and not _is_skipped_artifact(f)
                    and not _is_ephemeral_fact(f)]
        selected: List[Any] = []
        seen_ids: set = set()
        per_kind_count: Dict[str, int] = {}
        per_kind_keys: Dict[str, set] = {}
        for f in filtered:
            k = f.kind or ""
            if (k in self._SUMMARY_IDENTITY_KINDS
                    and per_kind_count.get(k, 0) < self._SUMMARY_IDENTITY_RESERVE):
                key_norm = (f.key or "").strip().lower()
                seen_keys = per_kind_keys.setdefault(k, set())
                if key_norm and key_norm in seen_keys:
                    continue
                val_low = (f.value or "").lower()
                if any(tok in val_low
                       for tok in self._SUMMARY_IDENTITY_VALUE_NOISE):
                    continue
                if key_norm:
                    seen_keys.add(key_norm)
                selected.append(f)
                seen_ids.add(id(f))
                per_kind_count[k] = per_kind_count.get(k, 0) + 1
        person_keys: set = set()
        person_count = 0
        for f in filtered:
            if id(f) in seen_ids:
                continue
            if (f.kind or "") != "person":
                continue
            key_norm = (f.key or "").strip().lower()
            if not key_norm or key_norm in person_keys:
                continue
            person_keys.add(key_norm)
            selected.append(f)
            seen_ids.add(id(f))
            person_count += 1
            if person_count >= self._SUMMARY_PERSON_RESERVE:
                break
        for f in filtered:
            if len(selected) >= max_facts:
                break
            if id(f) in seen_ids:
                continue
            selected.append(f)
            seen_ids.add(id(f))
        facts = selected[:max_facts]
        if not facts:
            return ""
        # Group by kind for readability. Sort within each group by most
        # recent first (find_facts already returns by ts desc, so order
        # is preserved when we iterate).
        by_kind: Dict[str, List[str]] = {}
        for f in facts:
            line = f"{f.key} = {f.value}"
            by_kind.setdefault(f.kind, []).append(line)
        parts: List[str] = []
        # Preferences first — they shape every interaction.
        kind_order = ["preference", "alias", "person", "place", "schedule",
                      "relation", "course", "interest", "role", "contact",
                      "fact", "artifact"]
        for kind in kind_order:
            if kind not in by_kind:
                continue
            entries = ", ".join(by_kind[kind][:20])
            parts.append(f"{kind}: {entries}")
        for kind, entries in by_kind.items():
            if kind not in kind_order:
                joined = ", ".join(entries[:20])
                parts.append(f"{kind}: {joined}")
        body = " | ".join(parts)
        if not body:
            return ""
        note = ("(Background context from prior conversations with this "
                f"user — use freely when relevant: {body})")
        return note[:max_chars]

    # ---- realtime conversation observation -------------------------------
    def record_conversational(self, user_text: str,
                              assistant_text: str = "") -> None:
        """Persist a realtime (non-planner) turn into the episodic store.

        Mirrors the episodic-write half of MemoryManager.record so that
        purely-conversational turns (no tool dispatch, no planner) still
        land in durable memory and are recoverable by recall() after a
        process restart. Without this, recall has no row to hit and the
        model confabulates from a near-empty match set.

        Honors incognito the same way SessionBuffer does (private-mode
        turns must NOT touch the durable store). Best-effort: never
        raises into the caller.
        """
        try:
            from ..incognito import is_incognito
            if is_incognito():
                if self._logger:
                    self._logger.event("memory_episodic_skip_incognito")
                return
        except Exception:
            pass
        ut = (user_text or "").strip()
        if not ut:
            return
        outcome = (assistant_text or "")[:600]
        if self._async:
            threading.Thread(
                target=self._episodic_insert_safe,
                args=(ut, None, None, outcome),
                name="iris-memory-write-realtime",
                daemon=True,
            ).start()
        else:
            self._episodic_insert_safe(ut, None, None, outcome)

    def observe_conversation(self, user_text: str,
                             assistant_text: str = "") -> None:
        """Persist a realtime turn episodically AND (optionally) extract
        durable facts from it. The episodic write happens unconditionally
        (subject to incognito) so recall can find this turn after a
        process restart. Fact extraction is gated by the LLM-extractor's
        own enabled/length checks.

        Skips short / question-only turns for fact extraction (where it
        would be pure noise — no factual content can plausibly be in
        'what's the weather'); the episodic row is still written."""
        # ALWAYS persist the turn episodically first — this is the fix
        # for the recall confabulation bug where realtime turns left no
        # durable trace. The previous behavior bailed entirely when
        # extraction was disabled, leaving recall to invent details.
        self.record_conversational(user_text, assistant_text)

        if not _llm_extraction_enabled():
            if self._logger:
                self._logger.event("memory_llm_skip_disabled")
            return
        ut = (user_text or "").strip()
        if len(ut) < 8:
            if self._logger:
                self._logger.event("memory_llm_skip_too_short", length=len(ut))
            return
        # Question-only turns rarely contain durable facts. Cheap guard
        # before paying for the LLM round-trip.
        if ut.endswith("?") and len(ut) < 50:
            if self._logger:
                self._logger.event("memory_llm_skip_short_question",
                                   length=len(ut))
            return
        if self._logger:
            self._logger.event("memory_llm_thread_starting",
                               user_text_len=len(ut))
        threading.Thread(
            target=self._observe_safe,
            args=(ut, (assistant_text or "").strip()),
            name="iris-fact-observe",
            daemon=True,
        ).start()

    def _observe_safe(self, user_text: str, assistant_text: str) -> None:
        # Draft-capture path: a draft request can be MULTI-TURN
        # ("draft an email" -> assistant: "what's going on?" -> user
        # adds detail -> assistant: <full body>). The same-turn gate
        # alone captured the clarifying question as the artifact, so
        # we add a pending-draft latch:
        #
        # State machine:
        #   1. User asks for a draft AND assistant delivered it the
        #      same turn -> record_chat_turn now, latch stays clear.
        #   2. User asks for a draft AND assistant replied with a
        #      clarifying question -> SUPPRESS the artifact write;
        #      arm the latch with the original (kind, topic, user_text).
        #   3. Latch armed AND a later assistant reply looks like a
        #      delivered draft -> record_chat_turn using the LATCHED
        #      user_text (so the artifact key still reflects the
        #      original topic), then clear the latch.
        #   4. Latch armed AND ttl_turns expires (6 turns) -> clear it
        #      to avoid stale promotion of unrelated long replies.
        #
        # All latch reads/writes are guarded by _pending_draft_lock
        # since _observe_safe runs on a worker thread.
        at_stripped = (assistant_text or "").strip()
        try:
            if _looks_like_draft_request(user_text) and at_stripped:
                kind = _classify_draft_kind(user_text)
                topic = _topic_slug_from_request(user_text)
                if _looks_like_delivered_draft(at_stripped, kind=kind):
                    # Same-turn delivery (case 1): write through, clear
                    # any prior latch so it can't double-fire.
                    self.record_chat_turn(user_text, assistant_text)
                    with self._pending_draft_lock:
                        self._pending_draft = None
                elif _looks_like_clarifying_question(at_stripped):
                    # Case 2: clarifying question — arm the latch,
                    # DO NOT write the bogus reply as an artifact.
                    with self._pending_draft_lock:
                        self._pending_draft = {
                            "kind": kind,
                            "topic": topic,
                            "user_text": user_text,
                            "ts": time.time(),
                            "ttl_turns": 6,
                        }
                    if self._logger:
                        self._logger.event("memory_draft_latch_armed",
                                           kind=kind, topic=topic or "")
                else:
                    # Reply is short/ambiguous and doesn't pattern-match a
                    # delivered draft. Skip artifact write to avoid the
                    # exact bug we're fixing; arm the latch so a later
                    # delivery turn can still be promoted.
                    with self._pending_draft_lock:
                        self._pending_draft = {
                            "kind": kind,
                            "topic": topic,
                            "user_text": user_text,
                            "ts": time.time(),
                            "ttl_turns": 6,
                        }
                    if self._logger:
                        self._logger.event("memory_draft_latch_armed_ambiguous",
                                           kind=kind, topic=topic or "",
                                           reply_chars=len(at_stripped))
            else:
                # Latch follow-up (case 3 / 4): a later turn whose
                # user_text didn't itself look drafty. If the assistant
                # reply now LOOKS like a delivered draft, promote it
                # using the latched intent. Otherwise tick the TTL.
                with self._pending_draft_lock:
                    pending = self._pending_draft
                if pending and at_stripped:
                    if _looks_like_delivered_draft(at_stripped,
                                                    kind=pending.get("kind")):
                        try:
                            self.record_chat_turn(
                                user_text or pending.get("user_text", ""),
                                assistant_text,
                                original_user_text=pending.get("user_text"),
                            )
                        finally:
                            with self._pending_draft_lock:
                                self._pending_draft = None
                        if self._logger:
                            self._logger.event(
                                "memory_draft_latch_promoted",
                                kind=pending.get("kind") or "",
                                topic=pending.get("topic") or "")
                    else:
                        # Tick TTL — drop the latch when it expires.
                        with self._pending_draft_lock:
                            if self._pending_draft is pending:
                                ttl = int(pending.get("ttl_turns", 0)) - 1
                                if ttl <= 0:
                                    self._pending_draft = None
                                    if self._logger:
                                        self._logger.event(
                                            "memory_draft_latch_expired",
                                            topic=pending.get("topic") or "")
                                else:
                                    pending["ttl_turns"] = ttl
        except Exception as exc:  # pragma: no cover - defensive
            if self._logger:
                self._logger.exception("memory_record_chat_turn_failed", exc)
        try:
            facts = extract_facts_from_conversation(
                user_text, assistant_text, logger=self._logger)
        except Exception as exc:  # pragma: no cover - defensive
            if self._logger:
                self._logger.exception("memory_llm_extract_failed", exc)
            return
        if not facts:
            if self._logger:
                # Distinguish 'API returned but parsed to zero' from earlier
                # silent-fail paths inside the extractor.
                self._logger.event("memory_llm_extract_empty",
                                   user_text_len=len(user_text))
            return
        threshold = _llm_confidence_threshold()
        kept = [(kind, key, value) for (kind, key, value, conf) in facts
                if conf >= threshold]
        if not kept:
            if self._logger:
                self._logger.event("memory_llm_extract_below_threshold",
                                   raw_count=len(facts),
                                   threshold=threshold)
            return
        # Provenance: LLM extraction is "conversation" — distinct from
        # planner-step extraction below and from explicit user_said
        # preferences. extracted_at captures when extraction completed.
        ext_at = time.time()
        for kind, key, value in kept:
            try:
                self._store.add_semantic(
                    kind, key, value,
                    source="realtime conversation",
                    source_kind="conversation",
                    extracted_at=ext_at,
                )
            except Exception as exc:  # pragma: no cover
                if self._logger:
                    self._logger.exception("memory_llm_extract_save_failed", exc)
        if self._logger:
            self._logger.event(
                "memory_llm_extract_saved",
                count=len(kept),
                facts=[{"kind": k, "key": ky, "value": v[:80]}
                        for k, ky, v in kept],
            )
        # Cortex bridge: every _PATTERN_CHECK_INTERVAL observations, scan
        # for repeated query habits and emit patternAdded for any pattern
        # we haven't seen before in this process. Cheap when the bridge
        # is None (we skip the embedder calls entirely). Failures are
        # logged but never escalated — pattern surfacing is decorative.
        try:
            self._maybe_emit_patterns()
        except Exception as exc:  # pragma: no cover - defensive
            if self._logger:
                self._logger.exception("memory_pattern_emit_failed", exc)

    # ---- one-off cleanup -------------------------------------------------
    def _purge_bogus_clarify_artifacts(self) -> int:
        """One-shot cleanup: delete llm_draft artifact rows whose value
        is actually a clarifying-question reply that the pre-latch gate
        captured by mistake. Targets rows where:

          - kind == 'artifact'
          - source_kind == 'llm_draft'
          - value starts with "I don't" / "I don't" / "I do not" /
            "I don't actually have" OR contains "need more details", AND
          - value is short (< 400 chars)

        Idempotent and cheap (one indexed read + a few deletes). Runs
        once at MemoryManager construction so a user inheriting the bad
        row gets a clean store without manual intervention. Returns the
        number of rows removed; mostly for tests / logging.
        """
        try:
            rows = self._store.list_facts_by_source("llm_draft", limit=500)
        except Exception:
            return 0
        bad_ids: List[int] = []
        # Use a tight set of prefix patterns plus a needle phrase that
        # was characteristic of the captured clarifying questions.
        bad_prefixes = (
            "i don't actually have",
            "i don’t actually have",
            "i don't have any details",
            "i don’t have any details",
            "i do not have any details",
            "i don't have the details",
            "i don’t have the details",
        )
        bad_needles = (
            "need more details",
            "need a bit more",
            "what's going on",
            "what’s going on",
            "give me a quick rundown",
        )
        for row in rows:
            v = (row.value or "").strip()
            if not v or len(v) >= 400:
                continue
            vlow = v.lower()
            if (any(vlow.startswith(p) for p in bad_prefixes)
                    or any(n in vlow for n in bad_needles)):
                bad_ids.append(row.id)
        if not bad_ids:
            return 0
        try:
            with self._store._lock, self._store._conn() as c:
                c.executemany(
                    "DELETE FROM semantic WHERE id = ?",
                    [(rid,) for rid in bad_ids],
                )
        except Exception:
            return 0
        if self._logger:
            self._logger.event("memory_purged_clarify_artifacts",
                               count=len(bad_ids))
        return len(bad_ids)

    def _maybe_emit_patterns(self) -> None:
        """Rate-limited pattern detection + cortex bridge emission.

        Runs every ``_PATTERN_CHECK_INTERVAL`` observations. Skips the
        analysis entirely when no cortex bridge is registered — there's
        no one to receive the signal, so the embedder calls inside
        find_repeated_patterns aren't worth paying for. New-pattern
        dedupe is in-memory only (per-process) so closing + reopening
        the simulator will re-surface known patterns once."""
        self._pattern_check_count += 1
        if self._pattern_check_count < _PATTERN_CHECK_INTERVAL:
            return
        self._pattern_check_count = 0
        # Lazy bridge lookup — avoid the import + work when no
        # simulator is running.
        try:
            from ..cortex.bridge import get_active_bridge
        except Exception:
            return
        bridge = get_active_bridge()
        if bridge is None:
            return
        try:
            from .patterns import find_repeated_patterns
            result = find_repeated_patterns(self._store, self._embedder, days=7)
        except Exception as exc:  # pragma: no cover - defensive
            if self._logger:
                self._logger.exception("memory_pattern_scan_failed", exc)
            return
        if result.get("cold_start"):
            return
        patterns = result.get("patterns") or []
        emitted = 0
        for pattern in patterns:
            label = (pattern.get("label") or "").strip()
            if not label or label in self._seen_pattern_labels:
                continue
            self._seen_pattern_labels.add(label)
            try:
                bridge.emit_pattern_added({
                    "label": label,
                    "kind": pattern.get("kind", "query"),
                    "count": int(pattern.get("count", 0)),
                })
                emitted += 1
            except Exception as exc:  # pragma: no cover - defensive
                if self._logger:
                    self._logger.exception("memory_pattern_emit_one_failed", exc)
            if emitted >= _PATTERN_EMIT_MAX:
                break
        if emitted and self._logger:
            self._logger.event("memory_pattern_emitted", count=emitted)

    # ---- write -----------------------------------------------------------
    def forget_fact(self, kind: Optional[str] = None,
                    key: Optional[str] = None,
                    value: Optional[str] = None) -> int:
        """Delete semantic rows matching the filters. Returns the count."""
        try:
            return self._store.delete_facts(kind=kind, key=key, value=value)
        except Exception as exc:  # pragma: no cover - defensive
            if self._logger:
                self._logger.exception("memory_forget_fact_failed", exc)
            return 0

    def set_fact(self, kind: str, key: str, value: str,
                 source: str = "user said",
                 source_kind: str = "user_said",
                 source_id: Optional[str] = None) -> None:
        """Persist a single semantic fact. Used by Tier 1 preference-setting
        commands ('always send from gmail') to write through to the store
        without going via the executor.

        Defaults source_kind='user_said' since direct preference-setting
        is always user-initiated. Callers from other paths should pass
        the right source_kind explicitly."""
        try:
            self._store.add_semantic(
                kind, key, value,
                source=source,
                source_kind=source_kind,
                source_id=source_id,
                extracted_at=time.time(),
            )
        except Exception as exc:  # pragma: no cover - defensive
            if self._logger:
                self._logger.exception("memory_set_fact_failed", exc)

    def record_chat_turn(self, user_text: str, assistant_text: str,
                         *, intent_hint: Optional[str] = None,
                         original_user_text: Optional[str] = None) -> None:
        """Persist a CHAT-only turn (no planner, no tool dispatch) that
        produced durable generated content. Writes two rows:

        1. An episodic row with the FULL assistant text as the outcome,
           so 'show me the email I drafted about the leak' can match via
           the embedding/cosine recall path after restart.
        2. A semantic 'artifact' row keyed `<kind>:<topic_slug>` (e.g.
           ``artifact email:leak``) so the cheaper substring/co-mention
           recall pass in recall() can surface the draft when the user
           later asks about it by topic.

        Gated by ``_looks_like_draft_request`` upstream — callers MUST
        check that gate before calling; this method does not re-gate so
        a deliberate planner-side call (with a planner-classified
        intent) can write through without retesting.

        ``original_user_text`` (optional): when the artifact is being
        written from a LATER assistant reply than the original draft
        request (the pending-draft-latch path), pass the original
        request text so the artifact key reflects the original topic
        ('email:leak-under-sink') rather than the delivery turn's text
        (which is usually a detail-clarification, not a topic phrase).
        """
        ut = (user_text or "").strip()
        at = (assistant_text or "").strip()
        if not ut or not at:
            return
        # Use the original draft request for kind+topic when supplied
        # (latched path); fall back to the current user_text otherwise.
        topic_source = (original_user_text or "").strip() or ut
        kind = ((intent_hint or "").strip().lower()
                or _classify_draft_kind(topic_source))
        topic = _topic_slug_from_request(topic_source)
        # Episodic write: full assistant text as outcome. Reuse the
        # existing async-or-sync episodic insert path so we don't
        # block the reply.
        outcome = at[:_OUTCOME_MAX_CHARS]
        if self._async:
            threading.Thread(
                target=self._episodic_insert_safe,
                args=(ut, None, None, outcome),
                name="iris-memory-chat-write",
                daemon=True,
            ).start()
        else:
            self._episodic_insert_safe(ut, None, None, outcome)
        # Semantic artifact write: short key so recall's substring match
        # (manager.py:557-587) hits on topic words. Value carries the
        # draft body itself (capped); source_kind='llm_draft' so the
        # summary filter can surface drafts without unbanning tool-link
        # artifacts.
        key = f"{kind}:{topic}" if topic else kind
        value = at[:_ARTIFACT_VALUE_MAX_CHARS]
        try:
            self._store.add_semantic(
                "artifact", key, value,
                source="llm draft",
                source_kind="llm_draft",
                extracted_at=time.time(),
            )
        except Exception as exc:  # pragma: no cover - defensive
            if self._logger:
                self._logger.exception("memory_chat_artifact_save_failed", exc)
        if self._logger:
            self._logger.event("memory_chat_turn_recorded",
                               kind=kind, topic=topic or "",
                               assistant_chars=len(at))

    def _record_llm_draft_artifact(self, user_text: str,
                                   message: str) -> None:
        """Planner-side helper: when the planner ran but produced a
        draft-shaped body without a tool link (e.g. outlook_compose
        built a message but didn't send), still file an artifact row
        so recall can find it by topic. The episodic row is already
        written by ``record()``, so this only writes the semantic
        artifact (no duplicate episode)."""
        ut = (user_text or "").strip()
        msg = (message or "").strip()
        if not ut or not msg:
            return
        kind = _classify_draft_kind(ut)
        topic = _topic_slug_from_request(ut)
        key = f"{kind}:{topic}" if topic else kind
        value = msg[:_ARTIFACT_VALUE_MAX_CHARS]
        self._store.add_semantic(
            "artifact", key, value,
            source="llm draft (planner)",
            source_kind="llm_draft",
            extracted_at=time.time(),
        )
        if self._logger:
            self._logger.event("memory_planner_draft_artifact_recorded",
                               kind=kind, topic=topic or "",
                               chars=len(msg))

    def find_llm_drafts_matching(self, user_text: str,
                                 k: int = 3) -> List[Any]:
        """Return up to ``k`` ``llm_draft`` semantic rows whose key suffix
        (after the leading ``<kind>:``) shares a non-stopword token with
        ``user_text``. Used by the realtime recall-context injector so a
        retrieval-shaped turn ('show me the email we drafted about the
        leak') gets the FULL draft body in the note — not just a 120-char
        truncated outcome.
        Bounded result count; safe to call when memory is empty (returns
        []). Reuses the same token-extraction shape as
        ``_topic_slug_from_request`` so matches stay consistent with
        write-side key construction."""
        if not user_text:
            return []
        k = max(1, int(k or 1))
        try:
            rows = self._store.list_facts_by_source("llm_draft", limit=200)
        except Exception:
            return []
        if not rows:
            return []
        # Token set from the user's text: lowercase alnum tokens with
        # stopwords stripped. Reuses _TOPIC_STOPWORDS so retrieval logic
        # mirrors what _topic_slug_from_request used at write time.
        toks: set = set()
        for raw in re.split(r"[^A-Za-z0-9'\-]+", user_text):
            w = raw.strip("-' ").lower()
            if not w or w in _TOPIC_STOPWORDS:
                continue
            if len(w) < 2:
                continue
            toks.add(w)
        if not toks:
            # No content tokens at all — fall back to most-recent drafts so
            # a generic 'show me the email we drafted' still surfaces the
            # latest one rather than nothing.
            return rows[:k]
        scored: List[tuple] = []
        for r in rows:
            key = (r.key or "").lower()
            # Strip the leading "<kind>:" so we match on the topic slug.
            if ":" in key:
                _, _, topic_part = key.partition(":")
            else:
                topic_part = key
            key_toks: set = set()
            for raw in re.split(r"[^A-Za-z0-9'\-]+", topic_part):
                w = raw.strip("-' ").lower()
                if not w or w in _TOPIC_STOPWORDS:
                    continue
                if len(w) < 2:
                    continue
                key_toks.add(w)
            overlap = len(toks & key_toks)
            if overlap > 0:
                scored.append((overlap, r))
        scored.sort(key=lambda x: x[0], reverse=True)
        matched = [r for _, r in scored[:k]]
        if matched:
            return matched
        # No topic-token overlap — fall back to the most-recent draft(s).
        # 'show me the email we drafted' with a generic topic-less key
        # ('email:' with empty topic) should still surface.
        return rows[:k]

    def record(self, user_text: str, plan: Any, steps: List[Any],
               results: List[Any], message: str) -> None:
        """Persist a planner-handled turn. Fact extraction is sync (fast);
        episodic embedding + insert runs on a worker thread when async_writes
        is True so the user reply isn't blocked by the embedder HTTP call."""
        facts = extract_facts(user_text, steps, results)
        if facts:
            try:
                # All facts from this batch came from planner step args
                # / results — tag accordingly so the cortex UI can group
                # 'things Iris learned by doing' separately from chat.
                self._store.add_facts(
                    facts,
                    source_kind="planner_step",
                    extracted_at=time.time(),
                )
            except Exception as exc:  # pragma: no cover - defensive
                if self._logger:
                    self._logger.exception("memory_add_facts_failed", exc)

        # Phase-3 wiring: implicit fact extraction from free-form
        # user text. Catches "I live in Berlin" / "my name is Dani" /
        # "always send via gmail" / etc. — patterns the planner-step
        # extractor doesn't see because they're not in the tool args.
        # Confidence-floored at 0.7 inside the extractor so noise
        # doesn't pollute memory. Best-effort.
        try:
            from ..implicit_facts import extract_implicit_facts
            from ..memory_pinning import maybe_pin_fact
            implicit = extract_implicit_facts(
                user_text=user_text,
                assistant_text=message,
                tool_results=None,
            )
            for cand in implicit:
                try:
                    self._store.add_semantic(
                        cand.kind, cand.key, cand.value,
                        source=cand.source,
                        source_kind=cand.source_kind,
                        source_id=None,
                        extracted_at=time.time(),
                    )
                except Exception:
                    continue
                # Phase-3: after each implicit write, check whether
                # this (kind, key) has been mentioned enough times
                # to auto-pin. Cheap; the pin function returns
                # False fast when below threshold.
                try:
                    maybe_pin_fact(store=self._store,
                                   kind=cand.kind, key=cand.key)
                except Exception:
                    continue
                # Phase-6: mirror person facts into the entity
                # graph so the pronoun resolver can bind 'him' /
                # 'her' / 'Dani' to a real entity. Best-effort.
                try:
                    if cand.kind == "person":
                        from ..entity_graph import (EntityKind,
                                                     global_graph)
                        global_graph().upsert_entity(
                            kind=EntityKind.PERSON.value,
                            name=cand.key.title(),
                            aliases=[cand.key],
                            attrs={"email": cand.value},
                        )
                except Exception:
                    continue
        except Exception as exc:  # pragma: no cover - defensive
            if self._logger:
                self._logger.exception(
                    "implicit_facts_failed", exc)

        plan_json = self._plan_to_json(plan)
        steps_json = self._steps_to_json(steps, results)
        # Cap is pure policy — column is unbounded TEXT. Old 600-char
        # limit chopped email/poem drafts mid-body; 8000 covers the
        # 99% case while still bounding pathological pastes.
        outcome = (message or "")[:_OUTCOME_MAX_CHARS]

        # Planner-routed draft (e.g. outlook_compose built a body but
        # no tool result returned a `link` because the message wasn't
        # actually sent): mirror the body into the artifact store so
        # 'show me the email I drafted' hits the same recall path the
        # chat-only branch uses below. Best-effort.
        try:
            if _looks_like_draft_request(user_text) and (message or "").strip():
                self._record_llm_draft_artifact(user_text, message)
        except Exception as exc:  # pragma: no cover - defensive
            if self._logger:
                self._logger.exception("memory_record_planner_draft_failed", exc)

        if self._async:
            threading.Thread(
                target=self._episodic_insert_safe,
                args=(user_text, plan_json, steps_json, outcome),
                name="iris-memory-write",
                daemon=True,
            ).start()
        else:
            self._episodic_insert_safe(user_text, plan_json, steps_json, outcome)

    # ---- read ------------------------------------------------------------
    def recall(self, goal_text: str, k: int = 3) -> Dict[str, Any]:
        """Return relevant memory for the goal. Episodic ranked by cosine
        similarity over embeddings; facts surfaced when any semantic key
        appears (substring) in the goal text."""
        goal = (goal_text or "").strip()
        episodes: List[Dict[str, Any]] = []
        facts: List[SemanticRow] = []
        project_context: List[Dict[str, Any]] = []
        if not goal:
            return {"episodes": episodes, "facts": facts,
                    "project_context": project_context, "context": ""}

        # ---- episodic: embed goal, cosine-rank against recent rows ------
        try:
            goal_vec = self._embedder.embed(goal)
        except Exception as exc:  # pragma: no cover - defensive
            if self._logger:
                self._logger.exception("memory_recall_embed_failed", exc)
            goal_vec = []
        if goal_vec:
            try:
                rows = self._store.list_episodic(limit=_RECALL_SCAN)
            except Exception:
                rows = []
            scored: List[tuple] = []
            for r in rows:
                if not r.embedding:
                    continue
                sim = cosine_sim(goal_vec, r.embedding)
                if sim >= _RECALL_MIN_SIM:
                    scored.append((sim, r))
            scored.sort(key=lambda x: x[0], reverse=True)
            for sim, r in scored[:k]:
                episodes.append({
                    "sim": round(sim, 3),
                    "ts": r.ts,
                    "user_text": r.user_text,
                    "outcome": r.outcome,
                })

        # ---- semantic: any fact whose key OR value appears in the goal --
        # Two passes: a direct (kind,key) / value substring match, then a
        # "co-mention" pass that pulls related facts. Without the second
        # pass, asking 'what's my dog's breed?' surfaces only key='dog'
        # (value=Benji) and never finds the row keyed 'pet-breed' because
        # 'pet-breed' isn't a phrase the user says.
        lower = goal.lower()
        try:
            candidates = self._store.find_facts(limit=200)
        except Exception:
            candidates = []
        seen_ids: set = set()
        primary_values: set = set()
        for row in candidates:
            hit = False
            # Artifact rows are keyed as '<kind>:<topic-slug>' (e.g.
            # 'email:leak-under-sink'). The full key never appears
            # verbatim in user queries, so split on ':' and try
            # kind + topic-tokens independently. The artifact hits when
            # BOTH (a) the kind word is in the goal, AND (b) at least
            # one topic token (>=3 chars) is in the goal — that pairing
            # keeps 'show me the email' from accidentally surfacing
            # every email-kind artifact in the store.
            if (row.kind == "artifact" and row.key and ":" in row.key):
                slug_kind, _, topic_slug = row.key.partition(":")
                slug_kind = slug_kind.strip()
                topic_tokens = [t for t in re.split(r"[^a-z0-9]+",
                                                     topic_slug.lower())
                                if len(t) >= 3]
                kind_hit = bool(
                    slug_kind and
                    re.search(rf"\b{re.escape(slug_kind)}\b", lower)
                )
                topic_hit = any(
                    re.search(rf"\b{re.escape(tok)}\b", lower)
                    for tok in topic_tokens
                )
                if kind_hit and (topic_hit or not topic_tokens):
                    hit = True
            if not hit and row.key and re.search(
                    rf"\b{re.escape(row.key)}\b", lower):
                hit = True
            elif not hit and row.value and len(row.value) >= 3:
                vlow = row.value.lower()
                if re.search(rf"\b{re.escape(vlow)}\b", lower):
                    hit = True
            if hit and row.id not in seen_ids:
                facts.append(row)
                seen_ids.add(row.id)
                if row.value:
                    primary_values.add(row.value.lower())
                if row.key:
                    primary_values.add(row.key.lower())
        # Generic 'draft' / 'drafted' / 'wrote' verb in the goal: surface
        # the MOST RECENT llm_draft artifact even when the goal doesn't
        # mention a topic word. Capped at 1 so this can't drown out a
        # topic-keyed hit when both fire. Keeps 'show me what we wrote'
        # answerable.
        _RECALL_GENERIC_DRAFT_RE = re.compile(
            r"\b(drafts?|drafted|drafting|wrote|written|composed?)\b",
            re.IGNORECASE,
        )
        if _RECALL_GENERIC_DRAFT_RE.search(lower):
            for row in candidates:
                if (row.kind == "artifact"
                        and (row.source_kind or "") == "llm_draft"
                        and row.id not in seen_ids):
                    facts.append(row)
                    seen_ids.add(row.id)
                    break
        if primary_values:
            for row in candidates:
                if row.id in seen_ids:
                    continue
                hay = f"{row.key or ''} {row.value or ''}".lower()
                if any(re.search(rf"\b{re.escape(token)}\b", hay)
                       for token in primary_values if token):
                    facts.append(row)
                    seen_ids.add(row.id)

        # ---- project RAG: hit any project whose label / id is named ----
        if self._project_store is not None:
            try:
                project_context = self._gather_project_context(goal)
            except Exception as exc:  # pragma: no cover - defensive
                if self._logger:
                    self._logger.exception("memory_recall_project_failed", exc)
                project_context = []

        # Cortex viz: pulse memory -> core and surface a leaf showing the
        # most-relevant hit (best-effort; silent when the cortex is closed).
        try:
            if episodes or facts or project_context:
                top_label: Optional[str] = None
                if project_context:
                    # Project hits get the most interesting viz, so
                    # surface them first when present.
                    pid = project_context[0].get("project_id") or ""
                    top_label = f"project: {pid}" if pid else None
                if not top_label and episodes:
                    txt = (episodes[0].get("user_text") or "").strip()
                    if txt:
                        top_label = txt[:48] + ("…" if len(txt) > 48 else "")
                if not top_label and facts:
                    top_label = getattr(facts[0], "key", None) or None
                cortex_emit.memory_retrieve(label=top_label)
        except Exception:
            pass

        return {
            "episodes": episodes,
            "facts": facts,
            "project_context": project_context,
            "context": self._render_context(episodes, facts, project_context),
        }

    # ---- project RAG helpers --------------------------------------------
    def _gather_project_context(self, goal: str) -> List[Dict[str, Any]]:
        """Match the goal text against registered project labels, then
        run a per-project semantic search. Returns a flat list of
        chunk dicts, capped at ``_RAG_TOP_K`` per project."""
        store = self._project_store
        if store is None:
            return []
        matches = store.match_projects_in_text(goal)
        broad = os.environ.get(_RAG_BROAD_ENV, "").strip() in ("1", "true", "yes")
        gathered: List[Dict[str, Any]] = []
        if matches:
            for proj in matches:
                pid = proj["project_id"]
                hits = store.search(pid, goal, k=_RAG_TOP_K)
                for h in hits:
                    gathered.append({
                        "project_id": pid,
                        "project_label": proj.get("label") or pid,
                        "file_path": h.file_path,
                        "sim": h.sim,
                        "chunk_text": h.chunk_text,
                    })
        elif broad and len(goal) >= 40:
            # Broad mode: search every project at a tighter threshold.
            hits = store.search_all(goal, k=_RAG_TOP_K)
            labels = {p["project_id"]: p.get("label") or p["project_id"]
                      for p in store.list_projects()}
            for h in hits:
                gathered.append({
                    "project_id": h.project_id,
                    "project_label": labels.get(h.project_id, h.project_id),
                    "file_path": h.file_path,
                    "sim": h.sim,
                    "chunk_text": h.chunk_text,
                })
        return gathered

    # ---- helpers --------------------------------------------------------
    def _episodic_insert_safe(self, user_text: str, plan_json: Optional[str],
                              steps_json: Optional[str], outcome: str) -> None:
        try:
            vec = self._embedder.embed(user_text)
        except Exception as exc:  # pragma: no cover
            if self._logger:
                self._logger.exception("memory_embed_failed", exc)
            vec = []
        try:
            self._store.add_episodic(user_text, plan_json, steps_json, outcome, vec)
        except Exception as exc:  # pragma: no cover
            if self._logger:
                self._logger.exception("memory_add_episodic_failed", exc)

    @staticmethod
    def _plan_to_json(plan: Any) -> Optional[str]:
        if plan is None:
            return None
        try:
            return json.dumps({
                "goal": getattr(plan, "goal", ""),
                "final": getattr(plan, "final", "return"),
                "n_steps": len(getattr(plan, "steps", []) or []),
            })
        except Exception:
            return None

    @staticmethod
    def _steps_to_json(steps: List[Any], results: List[Any]) -> Optional[str]:
        rows: List[Dict[str, Any]] = []
        for step, sr in zip(steps or [], results or []):
            try:
                rows.append({
                    "tool": getattr(step, "tool", "?"),
                    "status": getattr(sr, "status", "?"),
                })
            except Exception:
                continue
        try:
            return json.dumps(rows)
        except Exception:
            return None

    @staticmethod
    def _render_context(episodes: List[Dict[str, Any]],
                        facts: List[SemanticRow],
                        project_context: Optional[List[Dict[str, Any]]] = None
                        ) -> str:
        """Pre-render a "Context from prior turns" block plus any
        project-knowledge excerpts. Empty if there's nothing useful —
        caller can just check truthiness."""
        project_context = project_context or []
        if not episodes and not facts and not project_context:
            return ""
        chunks: List[str] = []

        if episodes or facts:
            lines: List[str] = ["Context from prior turns:"]
            for f in facts[:8]:
                lines.append(f"- {f.kind} {f.key} = {f.value}")
            for e in episodes:
                ut = (e.get("user_text") or "")[:120]
                oc = (e.get("outcome") or "")[:120]
                lines.append(f'- prior: "{ut}" -> {oc}')
            base = "\n".join(lines)
            if len(base) > _CONTEXT_MAX_CHARS:
                base = base[:_CONTEXT_MAX_CHARS] + "..."
            chunks.append(base)

        if project_context:
            # Group hits by project label so the prompt reads naturally.
            by_label: Dict[str, List[Dict[str, Any]]] = {}
            for item in project_context:
                lbl = item.get("project_label") or item.get("project_id") or "project"
                by_label.setdefault(lbl, []).append(item)
            rag_lines: List[str] = []
            for label, items in by_label.items():
                rag_lines.append(f"From your project {label}:")
                for it in items:
                    fname = os.path.basename(it.get("file_path") or "") or "?"
                    excerpt = (it.get("chunk_text") or "").strip()
                    excerpt = re.sub(r"\s+", " ", excerpt)[:200]
                    rag_lines.append(f"- {fname}: {excerpt}")
            rag_text = "\n".join(rag_lines)
            if len(rag_text) > _RAG_CONTEXT_MAX_CHARS:
                rag_text = rag_text[:_RAG_CONTEXT_MAX_CHARS] + "..."
            chunks.append(rag_text)

        return "\n\n".join(chunks)
