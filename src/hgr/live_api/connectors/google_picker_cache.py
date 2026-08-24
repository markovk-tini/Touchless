"""SQLite cache of user-picked Drive file_ids for the drive.file scope era.

Under the drive.file scope Google's Docs / Sheets / Slides APIs will only
let us reach a file when we already know its file_id AND that file was
either created by this app or opened via the Google Picker widget. This
cache remembers Picker choices so a user only has to pick "the Q4 budget"
once — every subsequent "append to Q4 budget" hits by slug lookup.

Storage: dedicated SQLite file under ~/Documents/Touchless/google/ so a
"Reset Google" wipe removes it in one go with the OAuth token. Override
via TOUCHLESS_GOOGLE_PICKER_CACHE for tests / dev.

Thread-safe via a per-instance lock; sqlite3 is opened with
check_same_thread=False so the connector ThreadPoolExecutors can read
without hopping to the GUI thread.

Author: Konstantin Markov
"""
from __future__ import annotations

import os
import re
import sqlite3
import threading
import time
from pathlib import Path
from typing import Dict, Iterable, Optional


# Reused from live_api/memory/manager.py — keep in sync there. Duplicated
# (not imported) so picker cache doesn't hard-depend on the memory package
# being importable in a dormant install.
_STOPWORDS = frozenset({
    "a", "an", "the", "my", "our", "your", "their", "his", "her", "its",
    "this", "that", "these", "those", "some", "any", "all",
    "to", "for", "of", "in", "on", "at", "by", "with", "about",
    "regarding", "concerning", "saying", "telling", "asking",
    "please", "kindly", "really", "very", "just",
    "and", "or", "but", "so",
    "today", "tomorrow", "tonight", "yesterday",
    # Verbs users tend to prepend when addressing an existing file:
    "add", "append", "put", "write", "insert", "update", "edit",
    "open", "show", "find", "get", "fetch",
    # Kind nouns — we key by kind separately, so drop them from the slug.
    "sheet", "sheets", "spreadsheet", "spreadsheets",
    "doc", "docs", "document", "documents",
    "slide", "slides", "slideshow", "presentation", "presentations",
    "file", "files",
})


_SCHEMA = """
CREATE TABLE IF NOT EXISTS picked_files (
    slug         TEXT NOT NULL,
    kind         TEXT NOT NULL,
    file_id      TEXT NOT NULL,
    name         TEXT,
    mime         TEXT,
    picked_at    REAL NOT NULL,
    last_used_at REAL,
    use_count    INTEGER DEFAULT 0,
    PRIMARY KEY (slug, kind)
);
CREATE INDEX IF NOT EXISTS idx_picked_kind_recent
    ON picked_files(kind, last_used_at DESC);
"""


def _default_db_path() -> Path:
    override = os.environ.get("TOUCHLESS_GOOGLE_PICKER_CACHE")
    if override:
        return Path(override).expanduser()
    base = Path.home() / "Documents" / "Touchless" / "google"
    return base / "picker_cache.sqlite3"


class PickerCache:
    """Thread-safe SQLite cache of user Picker choices, keyed by (slug, kind).

    Slug is the normalized user phrase (lower + stopword-stripped +
    kebab-cased); kind is 'sheet' | 'doc' | 'slide' (matches the Picker
    ViewId families).
    """

    def __init__(self, db_path: Optional[Path] = None) -> None:
        self._path = Path(db_path) if db_path else _default_db_path()
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._init_schema()

    # ---- schema / connection ----------------------------------------------
    def _conn(self) -> sqlite3.Connection:
        c = sqlite3.connect(
            str(self._path), check_same_thread=False, timeout=5.0)
        c.row_factory = sqlite3.Row
        return c

    def _init_schema(self) -> None:
        with self._lock, self._conn() as c:
            c.executescript(_SCHEMA)

    # ---- slugification ----------------------------------------------------
    @staticmethod
    def slugify(name_or_phrase: str) -> str:
        """Normalize a user phrase to a stable slug.

        "append to my Q4 budget" -> "q4-budget"
        "Q4 Budget"              -> "q4-budget"
        "the tp report"          -> "tp-report"
        Empty string on all-stopword input.
        """
        if not name_or_phrase:
            return ""
        tokens = []
        for raw in re.split(r"[^A-Za-z0-9'\-]+", str(name_or_phrase)):
            w = raw.strip("-' ").lower()
            if not w or w in _STOPWORDS:
                continue
            tokens.append(w)
            if len(tokens) >= 8:
                break
        return "-".join(tokens)

    # ---- reads ------------------------------------------------------------
    def lookup(self, user_query: str, kind: str) -> Optional[Dict[str, str]]:
        """Return {"file_id","name","mime"} for the best match, or None.

        Tries exact-slug first, then substring match on stored slugs so
        "the Q4 budget" hits a cached "q4-budget". Bumps last_used_at /
        use_count on a hit for future recency ranking.
        """
        slug = self.slugify(user_query)
        if not slug:
            return None
        kind = str(kind or "").strip().lower()
        with self._lock, self._conn() as c:
            row = c.execute(
                "SELECT file_id, name, mime FROM picked_files "
                "WHERE slug = ? AND kind = ?",
                (slug, kind),
            ).fetchone()
            if row is None:
                # Substring fallback — first try slugs that contain the
                # query, then slugs the query contains. ORDER BY
                # last_used_at DESC so a recently-used file wins ties.
                like = f"%{slug}%"
                row = c.execute(
                    "SELECT file_id, name, mime FROM picked_files "
                    "WHERE kind = ? AND (slug LIKE ? OR ? LIKE '%' || slug || '%') "
                    "ORDER BY last_used_at DESC LIMIT 1",
                    (kind, like, slug),
                ).fetchone()
            if row is None:
                return None
            now = time.time()
            c.execute(
                "UPDATE picked_files SET last_used_at = ?, "
                "use_count = COALESCE(use_count, 0) + 1 "
                "WHERE slug = ? AND kind = ?",
                (now, slug, kind),
            )
            return {"file_id": row["file_id"],
                    "name": row["name"] or "",
                    "mime": row["mime"] or ""}

    # ---- writes -----------------------------------------------------------
    def remember(self, user_query: str, kind: str,
                 file_id: str, name: str, mime: str) -> None:
        """Store a Picker choice keyed by BOTH the user's phrase AND the
        file's actual display name, so future lookups by either hit.
        No-op when file_id is empty."""
        if not file_id:
            return
        kind = str(kind or "").strip().lower()
        slugs: set = set()
        for src in (user_query, name):
            s = self.slugify(src)
            if s:
                slugs.add(s)
        if not slugs:
            return
        now = time.time()
        rows = [(s, kind, file_id, name or "", mime or "", now, now, 1)
                for s in slugs]
        with self._lock, self._conn() as c:
            c.executemany(
                "INSERT OR REPLACE INTO picked_files "
                "(slug, kind, file_id, name, mime, "
                " picked_at, last_used_at, use_count) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                rows,
            )

    def forget(self, user_query: str,
               kind: Optional[str] = None) -> int:
        """Remove cached entries so a "no, wrong file, pick again" flow can
        re-prompt. Returns number of rows deleted."""
        slug = self.slugify(user_query)
        if not slug:
            return 0
        with self._lock, self._conn() as c:
            if kind:
                cur = c.execute(
                    "DELETE FROM picked_files WHERE slug = ? AND kind = ?",
                    (slug, str(kind).strip().lower()),
                )
            else:
                cur = c.execute(
                    "DELETE FROM picked_files WHERE slug = ?", (slug,))
            return cur.rowcount or 0

    # ---- iteration (debug / diagnostics) ----------------------------------
    def all_rows(self) -> Iterable[Dict[str, str]]:
        with self._lock, self._conn() as c:
            for row in c.execute(
                    "SELECT slug, kind, file_id, name, mime, "
                    "       picked_at, last_used_at, use_count "
                    "FROM picked_files "
                    "ORDER BY last_used_at DESC"):
                yield dict(row)


_shared: Optional[PickerCache] = None
_shared_lock = threading.Lock()


def shared() -> PickerCache:
    """Module-level singleton. Mirrors GoogleClient.shared() so all
    connectors reuse one open DB handle."""
    global _shared
    with _shared_lock:
        if _shared is None:
            _shared = PickerCache()
        return _shared
