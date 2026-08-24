"""ProjectMemoryStore — RAG over user project folders.

Sibling DB to memory.db (kept separate so the heavy churn of source-code
chunks doesn't bloat the hot episodic table, and so the user can wipe
project knowledge independently).

Single table ``project_knowledge``:

  (id, project_id, file_path, chunk_index, chunk_text, embedding,
   token_count, file_mtime, file_size, last_indexed_at)

Embeddings stored as raw float32 BLOBs (reusing the same encoding the
episodic store uses, so we get to share ``_encode_vec``/``_decode_vec``).

The store is thread-safe (per-call connection + per-instance lock for
writes) and intentionally does ALL its work synchronously — callers
that want background indexing wrap ``index_project`` in a thread (the
indexer's existing pattern, mirrors MemoryManager.observe_conversation).

Usage::

    from hgr.live_api.memory import ProjectMemoryStore, default_embedder
    pms = ProjectMemoryStore(embedder=default_embedder())
    pms.register_project("my-repo", Path(r"C:\\repo"), label="My Repo")
    pms.index_project("my-repo")  # walks + chunks + embeds
    hits = pms.search("my-repo", "how does the orchestrator dispatch?")

CLI smoke test at the bottom of the file (``python project_memory.py``).

Author: Konstantin Markov
"""
from __future__ import annotations

import os
import re
import sqlite3
import struct
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .embedder import Embedder, cosine_sim, default_embedder
from .store import _decode_vec, _encode_vec


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Files larger than this are skipped (likely minified or data dumps).
_MAX_FILE_BYTES = 100 * 1024  # 100 KB per spec

# Per-project hard chunk cap; protects against runaway prose repos.
_MAX_CHUNKS_PER_PROJECT = int(os.environ.get("TOUCHLESS_RAG_MAX_CHUNKS", "8000"))

# Char-based chunking (simpler, no tokenizer dep). ~500 tokens at
# 4 chars/token = ~2000 chars; ~80 tokens overlap = ~320 chars. But
# the design spec says "chars not tokens", so we use char counts
# directly: 500 / 80.
_CHUNK_CHARS = 500
_CHUNK_OVERLAP = 80

# Allowlist of file extensions to index. Anything else is skipped.
_INDEX_EXTENSIONS = frozenset({
    ".md", ".py", ".html", ".css", ".js", ".ts", ".tsx", ".jsx",
    ".json", ".yaml", ".yml", ".toml",
    # Plain text variants — useful for design docs.
    ".txt", ".rst",
})

# Directories never descended into.
_SKIP_DIRS = frozenset({
    ".git", "node_modules", "__pycache__", ".venv", "venv", "env",
    "dist", "build", ".next", "target", ".cache", ".idea", ".vscode",
    "site-packages", ".pytest_cache", ".mypy_cache", ".tox",
})

# Minimum cosine similarity for a chunk to be considered "relevant" in
# the search pathway.
_SEARCH_MIN_SIM = 0.18

# Minimum cosine similarity for the broad ("search all projects") path.
# Tighter because the candidate set is much larger and we want fewer
# false positives.
_SEARCH_BROAD_MIN_SIM = 0.30


# ---------------------------------------------------------------------------
# Path helper
# ---------------------------------------------------------------------------

def default_project_memory_path() -> Path:
    """Same scheme as ``default_memory_path`` in manager.py, but with a
    distinct filename so the project DB lives next to the episodic DB."""
    override = os.environ.get("TOUCHLESS_PROJECT_MEMORY_DB")
    if override:
        return Path(override)
    if os.name == "nt":
        base = os.environ.get("LOCALAPPDATA") or str(Path.home())
        return Path(base) / "Touchless" / "project_memory.db"
    return Path.home() / ".touchless" / "project_memory.db"


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_SCHEMA = """
CREATE TABLE IF NOT EXISTS project_knowledge (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id      TEXT NOT NULL,
    file_path       TEXT NOT NULL,
    chunk_index     INTEGER NOT NULL,
    chunk_text      TEXT NOT NULL,
    embedding       BLOB NOT NULL,
    token_count     INTEGER,
    file_mtime      REAL NOT NULL,
    file_size       INTEGER,
    last_indexed_at REAL NOT NULL,
    UNIQUE(file_path, chunk_index) ON CONFLICT REPLACE
);
CREATE INDEX IF NOT EXISTS idx_pk_project ON project_knowledge(project_id);
CREATE INDEX IF NOT EXISTS idx_pk_path    ON project_knowledge(file_path);

CREATE TABLE IF NOT EXISTS project_registry (
    project_id      TEXT PRIMARY KEY,
    label           TEXT NOT NULL,
    root_path       TEXT NOT NULL,
    registered_at   REAL NOT NULL
);
"""


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ProjectChunk:
    project_id: str
    file_path: str
    chunk_index: int
    chunk_text: str
    sim: float = 0.0  # Populated by search()


@dataclass
class IndexReport:
    project_id: str
    root_path: str
    files_scanned: int = 0
    files_indexed: int = 0
    files_skipped: int = 0
    chunks_written: int = 0
    chunks_skipped_mtime: int = 0
    errors: int = 0
    truncated: bool = False  # True iff we hit _MAX_CHUNKS_PER_PROJECT


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------

def _looks_binary(sample: bytes) -> bool:
    """Heuristic: more than 5% non-printable bytes in the first chunk
    means the file is probably binary (PDFs, images, minified blobs)."""
    if not sample:
        return False
    nontext = 0
    for b in sample:
        # Allow common text bytes: tab, LF, CR, plus printable ASCII +
        # high bytes (UTF-8 multibyte).
        if b in (0x09, 0x0A, 0x0D):
            continue
        if 0x20 <= b <= 0x7E:
            continue
        if b >= 0x80:  # likely UTF-8 continuation
            continue
        nontext += 1
    return (nontext / len(sample)) > 0.05


def _split_paragraphs(text: str) -> List[str]:
    # Split on blank lines (paragraph boundary).
    parts = re.split(r"\n\s*\n", text)
    return [p for p in parts if p.strip()]


def chunk_text(text: str,
               size: int = _CHUNK_CHARS,
               overlap: int = _CHUNK_OVERLAP) -> List[str]:
    """Split ``text`` into ~``size``-char overlapping chunks.

    Prefers paragraph boundaries, then sentence boundaries; falls back
    to hard char-split so we never emit zero-byte chunks. Pure function
    (no I/O) — easy to unit test.
    """
    text = (text or "").strip()
    if not text:
        return []
    if len(text) <= size:
        return [text]
    if overlap >= size:
        overlap = max(0, size // 4)

    # First, try to assemble chunks paragraph-by-paragraph.
    paragraphs = _split_paragraphs(text)
    chunks: List[str] = []
    buf = ""
    for p in paragraphs:
        if not buf:
            buf = p
            continue
        if len(buf) + 2 + len(p) <= size:
            buf = buf + "\n\n" + p
        else:
            chunks.append(buf)
            buf = p
    if buf:
        chunks.append(buf)

    # Second pass: any chunk still bigger than `size` is hard-split with
    # overlap. This handles single huge paragraphs (minified JSON, long
    # log lines, etc).
    final: List[str] = []
    for c in chunks:
        if len(c) <= size:
            final.append(c)
            continue
        # Hard window with overlap.
        step = max(1, size - overlap)
        i = 0
        while i < len(c):
            piece = c[i:i + size]
            if piece.strip():
                final.append(piece)
            if i + size >= len(c):
                break
            i += step

    return [c for c in final if c.strip()]


def _safe_read_text(path: Path) -> Optional[str]:
    """Read a file as text. Returns None when:
      - it can't be opened,
      - it appears binary,
      - it exceeds ``_MAX_FILE_BYTES``.
    """
    try:
        size = path.stat().st_size
    except OSError:
        return None
    if size > _MAX_FILE_BYTES:
        return None
    try:
        with open(path, "rb") as fh:
            sample = fh.read(4096)
            if _looks_binary(sample):
                return None
            rest = fh.read()
        raw = sample + rest
    except OSError:
        return None
    # Try UTF-8 first, fall back to latin-1 (lossless byte mapping).
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        try:
            return raw.decode("latin-1")
        except Exception:  # pragma: no cover - latin-1 never fails
            return None


def _iter_indexable_files(root: Path) -> Iterable[Path]:
    for dirpath, dirnames, filenames in os.walk(str(root)):
        # Mutate dirnames in place so os.walk doesn't descend into them.
        dirnames[:] = [d for d in dirnames if d not in _SKIP_DIRS
                       and not d.startswith(".")]
        for name in filenames:
            ext = os.path.splitext(name)[1].lower()
            if ext not in _INDEX_EXTENSIONS:
                continue
            yield Path(dirpath) / name


def _tokenize_for_match(text: str) -> List[str]:
    """Lowercase alphanumeric token split — used to match project labels
    against a user goal."""
    return re.findall(r"[a-z0-9]+", (text or "").lower())


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------

class ProjectMemoryStore:
    """SQLite-backed project knowledge store + RAG operations.

    Wraps both the schema/CRUD and the (synchronous) indexer. The
    indexer is intentionally simple: walk → filter → chunk → embed →
    upsert. Callers can run ``index_project`` on a thread; the store
    itself is thread-safe.
    """

    def __init__(self,
                 db_path: Optional[Path] = None,
                 embedder: Optional[Embedder] = None,
                 logger: Any = None) -> None:
        self._path = Path(db_path) if db_path else default_project_memory_path()
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._embedder = embedder or default_embedder()
        self._logger = logger
        self._init_schema()

    # ---- low-level ----
    def _conn(self) -> sqlite3.Connection:
        c = sqlite3.connect(str(self._path), check_same_thread=False, timeout=5.0)
        c.row_factory = sqlite3.Row
        return c

    def _init_schema(self) -> None:
        with self._lock, self._conn() as c:
            c.executescript(_SCHEMA)

    # ---- registry ----
    def register_project(self, project_id: str, root_path: Path,
                         *, label: Optional[str] = None) -> None:
        """Record this project's existence in the registry. Idempotent —
        re-registering updates the label / root."""
        if not project_id:
            raise ValueError("project_id required")
        root_abs = str(Path(root_path).resolve())
        lbl = (label or Path(root_abs).name or project_id).strip()
        with self._lock, self._conn() as c:
            c.execute(
                "INSERT OR REPLACE INTO project_registry "
                "(project_id, label, root_path, registered_at) "
                "VALUES (?, ?, ?, ?)",
                (project_id, lbl, root_abs, time.time()),
            )

    def list_projects(self) -> List[Dict[str, Any]]:
        with self._conn() as c:
            rows = c.execute(
                "SELECT project_id, label, root_path, registered_at "
                "FROM project_registry ORDER BY registered_at DESC"
            ).fetchall()
        return [
            {
                "project_id": r["project_id"],
                "label": r["label"],
                "root_path": r["root_path"],
                "registered_at": float(r["registered_at"]),
            }
            for r in rows
        ]

    def get_project(self, project_id: str) -> Optional[Dict[str, Any]]:
        with self._conn() as c:
            r = c.execute(
                "SELECT project_id, label, root_path, registered_at "
                "FROM project_registry WHERE project_id = ?",
                (project_id,),
            ).fetchone()
        if r is None:
            return None
        return {
            "project_id": r["project_id"],
            "label": r["label"],
            "root_path": r["root_path"],
            "registered_at": float(r["registered_at"]),
        }

    # ---- chunk-level writes / deletes ----
    def _delete_file(self, file_path: str) -> int:
        with self._lock, self._conn() as c:
            cur = c.execute(
                "DELETE FROM project_knowledge WHERE file_path = ?",
                (file_path,),
            )
            return cur.rowcount or 0

    def clear_project(self, project_id: str) -> int:
        """Drop all chunks for a project + its registry entry. Returns
        the number of chunks removed."""
        with self._lock, self._conn() as c:
            cur = c.execute(
                "DELETE FROM project_knowledge WHERE project_id = ?",
                (project_id,),
            )
            removed = cur.rowcount or 0
            c.execute(
                "DELETE FROM project_registry WHERE project_id = ?",
                (project_id,),
            )
            return int(removed)

    def count(self, project_id: Optional[str] = None) -> int:
        with self._conn() as c:
            if project_id:
                r = c.execute(
                    "SELECT COUNT(*) FROM project_knowledge WHERE project_id = ?",
                    (project_id,),
                ).fetchone()
            else:
                r = c.execute("SELECT COUNT(*) FROM project_knowledge").fetchone()
        return int(r[0])

    def _max_mtime_for_file(self, file_path: str) -> Optional[float]:
        with self._conn() as c:
            r = c.execute(
                "SELECT MAX(file_mtime) AS m FROM project_knowledge "
                "WHERE file_path = ?",
                (file_path,),
            ).fetchone()
        if r is None or r["m"] is None:
            return None
        return float(r["m"])

    def _insert_chunks(self,
                       project_id: str,
                       file_path: str,
                       file_mtime: float,
                       file_size: int,
                       chunks: Sequence[Tuple[str, List[float]]]) -> int:
        """Insert chunks for a file. Caller must have already deleted
        any prior chunks for this file_path. Returns rows written."""
        if not chunks:
            return 0
        now = time.time()
        rows = []
        for idx, (text, vec) in enumerate(chunks):
            rows.append((
                project_id, file_path, idx, text,
                _encode_vec(vec), len(text) // 4,
                file_mtime, file_size, now,
            ))
        with self._lock, self._conn() as c:
            c.executemany(
                "INSERT OR REPLACE INTO project_knowledge "
                "(project_id, file_path, chunk_index, chunk_text, "
                " embedding, token_count, file_mtime, file_size, "
                " last_indexed_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                rows,
            )
        return len(rows)

    # ---- indexer ----
    def index_project(self, project_id: str,
                      *, force: bool = False) -> IndexReport:
        """Walk the project's registered root, chunk + embed every
        indexable file, upsert. Skips files whose mtime hasn't changed
        since the last index (unless ``force=True``)."""
        proj = self.get_project(project_id)
        if proj is None:
            raise KeyError(f"project not registered: {project_id}")
        root = Path(proj["root_path"])
        report = IndexReport(project_id=project_id, root_path=str(root))
        if not root.is_dir():
            if self._logger:
                self._logger.event("project_rag_root_missing", project_id=project_id,
                                   root=str(root))
            return report

        for path in _iter_indexable_files(root):
            report.files_scanned += 1
            if report.chunks_written >= _MAX_CHUNKS_PER_PROJECT:
                report.truncated = True
                break
            file_abs = str(path.resolve())
            try:
                mtime = path.stat().st_mtime
                size = path.stat().st_size
            except OSError:
                report.files_skipped += 1
                continue

            if not force:
                prev = self._max_mtime_for_file(file_abs)
                if prev is not None and mtime <= prev:
                    report.chunks_skipped_mtime += 1
                    continue

            text = _safe_read_text(path)
            if text is None or not text.strip():
                report.files_skipped += 1
                continue
            pieces = chunk_text(text)
            if not pieces:
                report.files_skipped += 1
                continue

            # Embed every chunk; skip the file entirely on embed failure
            # (don't poison the index with a partial file).
            embedded: List[Tuple[str, List[float]]] = []
            file_failed = False
            for piece in pieces:
                try:
                    vec = self._embedder.embed(piece)
                except Exception as exc:  # pragma: no cover - defensive
                    if self._logger:
                        self._logger.exception("project_rag_embed_failed", exc)
                    file_failed = True
                    report.errors += 1
                    break
                if not vec:
                    file_failed = True
                    break
                embedded.append((piece, vec))
            if file_failed or not embedded:
                report.files_skipped += 1
                continue

            # Cap respect: if this file would put us over, trim.
            remaining = _MAX_CHUNKS_PER_PROJECT - report.chunks_written
            if remaining <= 0:
                report.truncated = True
                break
            if len(embedded) > remaining:
                embedded = embedded[:remaining]
                report.truncated = True

            # Drop prior chunks for this file (simpler than diffing).
            self._delete_file(file_abs)
            wrote = self._insert_chunks(
                project_id=project_id,
                file_path=file_abs,
                file_mtime=mtime,
                file_size=size,
                chunks=embedded,
            )
            report.chunks_written += wrote
            report.files_indexed += 1
        return report

    def index_file(self, project_id: str, path: Path,
                   *, force: bool = False) -> int:
        """Single-file convenience index. Returns chunks written."""
        path = Path(path)
        if not path.is_file():
            return 0
        file_abs = str(path.resolve())
        try:
            mtime = path.stat().st_mtime
            size = path.stat().st_size
        except OSError:
            return 0
        if not force:
            prev = self._max_mtime_for_file(file_abs)
            if prev is not None and mtime <= prev:
                return 0
        text = _safe_read_text(path)
        if text is None:
            return 0
        pieces = chunk_text(text)
        if not pieces:
            return 0
        embedded: List[Tuple[str, List[float]]] = []
        for piece in pieces:
            try:
                vec = self._embedder.embed(piece)
            except Exception:
                return 0
            if not vec:
                return 0
            embedded.append((piece, vec))
        self._delete_file(file_abs)
        return self._insert_chunks(project_id, file_abs, mtime, size, embedded)

    # ---- search ----
    def search(self, project_id: str, query_text: str,
               k: int = 4,
               min_sim: float = _SEARCH_MIN_SIM) -> List[ProjectChunk]:
        """Top-K most similar chunks for a single project."""
        query = (query_text or "").strip()
        if not query:
            return []
        try:
            qvec = self._embedder.embed(query)
        except Exception as exc:  # pragma: no cover - defensive
            if self._logger:
                self._logger.exception("project_rag_search_embed_failed", exc)
            return []
        if not qvec:
            return []
        with self._conn() as c:
            rows = c.execute(
                "SELECT project_id, file_path, chunk_index, chunk_text, embedding "
                "FROM project_knowledge WHERE project_id = ?",
                (project_id,),
            ).fetchall()
        return self._rank_rows(rows, qvec, k, min_sim)

    def search_all(self, query_text: str, k: int = 4,
                   project_ids: Optional[Sequence[str]] = None,
                   min_sim: Optional[float] = None) -> List[ProjectChunk]:
        """Top-K across one or more projects. When ``project_ids`` is
        None, searches every chunk in the DB (used by 'broad' mode in
        the manager)."""
        query = (query_text or "").strip()
        if not query:
            return []
        try:
            qvec = self._embedder.embed(query)
        except Exception:
            return []
        if not qvec:
            return []
        sql = ("SELECT project_id, file_path, chunk_index, chunk_text, embedding "
               "FROM project_knowledge")
        args: List[Any] = []
        if project_ids:
            placeholders = ",".join("?" for _ in project_ids)
            sql += f" WHERE project_id IN ({placeholders})"
            args.extend(project_ids)
        with self._conn() as c:
            rows = c.execute(sql, args).fetchall()
        if min_sim is None:
            min_sim = (_SEARCH_BROAD_MIN_SIM if project_ids is None
                       else _SEARCH_MIN_SIM)
        return self._rank_rows(rows, qvec, k, min_sim)

    @staticmethod
    def _rank_rows(rows: Sequence[sqlite3.Row], qvec: List[float],
                   k: int, min_sim: float) -> List[ProjectChunk]:
        scored: List[Tuple[float, sqlite3.Row]] = []
        for r in rows:
            vec = _decode_vec(r["embedding"])
            if not vec:
                continue
            sim = cosine_sim(qvec, vec)
            if sim >= min_sim:
                scored.append((sim, r))
        scored.sort(key=lambda x: x[0], reverse=True)
        out: List[ProjectChunk] = []
        for sim, r in scored[: max(0, int(k))]:
            out.append(ProjectChunk(
                project_id=str(r["project_id"]),
                file_path=str(r["file_path"]),
                chunk_index=int(r["chunk_index"]),
                chunk_text=str(r["chunk_text"]),
                sim=round(float(sim), 4),
            ))
        return out

    # ---- public convenience -------------------------------------------
    def match_projects_in_text(self, text: str) -> List[Dict[str, Any]]:
        """Return registered projects whose label OR id token appears
        as a whole word in ``text``. Used by MemoryManager.recall to
        decide which projects to search."""
        tokens = set(_tokenize_for_match(text))
        if not tokens:
            return []
        out: List[Dict[str, Any]] = []
        for proj in self.list_projects():
            label_tokens = set(_tokenize_for_match(proj["label"]))
            id_tokens = set(_tokenize_for_match(proj["project_id"]))
            # We want a *meaningful* hit — single-letter / common words
            # like "a", "the" would false-positive. Require token length
            # >= 3 to count.
            candidates = {t for t in (label_tokens | id_tokens) if len(t) >= 3}
            if candidates & tokens:
                out.append(proj)
        return out


# ---------------------------------------------------------------------------
# __main__ smoke test
# ---------------------------------------------------------------------------

def _smoke_test() -> int:
    """Build a tiny fake project, index it, search it, assert hits.
    Prints PASS / FAIL and returns 0/1 for shell."""
    import shutil
    import tempfile

    from .embedder import FakeEmbedder

    tmp = Path(tempfile.mkdtemp(prefix="iris-pms-smoke-"))
    db = tmp / "smoke.db"
    root = tmp / "fake-project"
    (root / "docs").mkdir(parents=True)
    (root / "src").mkdir(parents=True)

    (root / "README.md").write_text(
        "# Fake Project\n\nThis project demonstrates the orchestrator "
        "pipeline. The orchestrator dispatches user goals to planner "
        "tools. Memory recall is invoked first.\n",
        encoding="utf-8",
    )
    (root / "docs" / "architecture.md").write_text(
        "# Architecture\n\nWe use SQLite for episodic memory and a "
        "separate sqlite file for project knowledge. The embedder is "
        "pluggable.\n",
        encoding="utf-8",
    )
    (root / "src" / "main.py").write_text(
        "def run():\n    # entry point for the fake project pipeline\n"
        "    return 'ok'\n",
        encoding="utf-8",
    )
    # A binary-looking file that should be skipped.
    (root / "blob.json").write_bytes(b"\x00\x01\x02" * 100)
    # A too-large file that should be skipped.
    (root / "big.md").write_text("x" * (_MAX_FILE_BYTES + 10), encoding="utf-8")

    failures: List[str] = []
    try:
        pms = ProjectMemoryStore(db_path=db, embedder=FakeEmbedder())
        pms.register_project("fake-project", root, label="Fake Project")
        report = pms.index_project("fake-project")

        if report.files_indexed < 3:
            failures.append(
                f"expected >=3 files indexed, got {report.files_indexed}"
            )
        if report.chunks_written < 3:
            failures.append(
                f"expected >=3 chunks written, got {report.chunks_written}"
            )
        # blob.json (binary) and big.md (oversize) should NOT have
        # produced chunks; they should be counted as skipped.
        if report.files_skipped < 2:
            failures.append(
                "expected at least 2 files skipped (binary+oversize), "
                f"got {report.files_skipped}"
            )

        # Re-index without --force: every file should skip on mtime.
        report2 = pms.index_project("fake-project")
        if report2.chunks_written != 0:
            failures.append(
                f"re-index wrote {report2.chunks_written} chunks; expected 0"
            )
        if report2.chunks_skipped_mtime < 3:
            failures.append(
                f"expected mtime-skip >=3, got {report2.chunks_skipped_mtime}"
            )

        # Search for content that appears in README.
        hits = pms.search("fake-project", "orchestrator pipeline planner")
        if not hits:
            failures.append("expected at least one search hit for 'orchestrator'")
        else:
            joined = " ".join(h.chunk_text.lower() for h in hits)
            if "orchestrator" not in joined and "planner" not in joined:
                failures.append(
                    "top hit text did not contain expected keywords; "
                    f"got: {joined[:200]!r}"
                )

        # match_projects_in_text — token in text picks up the project.
        matches = pms.match_projects_in_text(
            "tell me about the fake project orchestrator"
        )
        if not matches:
            failures.append("match_projects_in_text returned no projects")

        # clear_project should remove chunks AND registry row.
        removed = pms.clear_project("fake-project")
        if removed < 3:
            failures.append(f"clear_project removed only {removed} chunks")
        if pms.get_project("fake-project") is not None:
            failures.append("clear_project did not remove the registry row")

    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    if failures:
        print("FAIL")
        for f in failures:
            print("  -", f)
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(_smoke_test())
