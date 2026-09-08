"""Contacts gatherer — aggregate person/email knowledge from every
source iris currently has access to:

    1. memory.db semantic facts with kind in ('person', 'contact')
    2. memory.db episodic mentions (proper-noun + email regex sweep)
    3. MS365 contacts_search (Graph: /me/contacts + /me/people +
       /me/messages header mining) across every connected MS account
    4. Gmail senders extracted from gmail_list headers
    5. MS365 calendar attendees (/me/events expanded one-by-one for the
       first N upcoming/recent events)
    6. Per-project git log authors (`git log --format=%aN <email>`)

Returns a deduplicated, ranked list shaped for the cortex tier-3
contacts leaf. Never crashes the caller on missing auth / missing
connector / broken DB: each source is independently try/excepted, and
missing sources just drop out silently. Each contribution is capped
before the global dedup so a single chatty source can't drown out the
rest.

Author: Konstantin Markov
"""
from __future__ import annotations

import concurrent.futures
import json
import os
import re
import subprocess
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple


# Hard per-source HTTP timeout. MS365 Graph + Gmail calls can stall
# indefinitely on a degraded network or a stale token mid-refresh; this
# bounds each source so a single slow connector can't add minutes to the
# whole simulator launch. Override via env for tests / dev.
_SOURCE_TIMEOUT_S = float(os.environ.get("TOUCHLESS_CONTACTS_SOURCE_TIMEOUT", "3.0"))


def _run_source_with_timeout(fn: Callable[[], int],
                             label: str,
                             timeout_s: float = _SOURCE_TIMEOUT_S) -> int:
    """Run a _from_* helper on a worker thread bounded by ``timeout_s``.
    On timeout / exception, log a warning and return 0 (no contacts
    added from this source). Never raises."""
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
            fut = ex.submit(fn)
            try:
                return int(fut.result(timeout=timeout_s) or 0)
            except concurrent.futures.TimeoutError:
                print(
                    f"WARN: contacts source {label} timed out after "
                    f"{timeout_s:.1f}s; continuing without it",
                    file=sys.stderr,
                )
                return 0
            except Exception as exc:
                print(
                    f"WARN: contacts source {label} crashed: {exc}",
                    file=sys.stderr,
                )
                return 0
    except Exception as exc:
        print(f"WARN: contacts source {label} executor failed: {exc}",
              file=sys.stderr)
        return 0


# Per-source confidence weights. Factual stores (semantic 'person'
# facts, MS365 contacts) trump heuristic mentions; git-config and
# episodic mentions sit lower.
_CONFIDENCE = {
    "memory.person": 1.0,
    "ms365.contacts": 0.95,
    "gmail.sender": 0.85,
    "ms365.calendar": 0.85,
    "git.config": 0.70,
    "episodic.mention": 0.65,
}

# Per-source contribution cap. Prevents one chatty source (e.g. 200
# gmail headers) from dominating the result before global dedup.
_PER_SOURCE_CAP = 30

# Domains we never want surfaced as real contacts (test/placeholder).
_PLACEHOLDER_DOMAINS = {
    "example.com", "example.org", "example.net",
    "test.com", "test.org", "localhost",
    "no-reply.com", "noreply.com", "donotreply.com",
}

# Email regex — same shape as memory.extractor._EMAIL_RE.
_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")

# Proper noun heuristic for episodic name extraction. Capitalized word
# 2+ letters long; we'll only accept pairs that co-occur in the same
# episode text, so single false positives don't pollute the output.
_NAME_RE = re.compile(r"\b[A-Z][a-zA-Z]{1,}\b")

# Common English capitalized words that aren't names. Filtered out so
# "Yes, I emailed dani@x.com" doesn't add 'Yes' as a contact.
_NAME_STOPWORDS = frozenset({
    "I", "The", "A", "An", "This", "That", "These", "Those",
    "Yes", "No", "Ok", "Okay", "OK", "Sure", "Maybe",
    "Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun",
    "Monday", "Tuesday", "Wednesday", "Thursday", "Friday",
    "Saturday", "Sunday",
    "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug",
    "Sep", "Oct", "Nov", "Dec",
    "January", "February", "March", "April", "June", "July",
    "August", "September", "October", "November", "December",
    "Iris", "Touchless", "Claude", "GPT", "Gmail", "Outlook",
    "Hello", "Hi", "Hey", "Thanks", "Thank", "Please",
    "Microsoft", "Google", "Apple", "Amazon",
    "Subject", "From", "To", "Cc", "Bcc", "Re", "Fwd",
})


def _is_placeholder_email(email: str) -> bool:
    if not email or "@" not in email:
        return True
    domain = email.rsplit("@", 1)[-1].lower().strip()
    return domain in _PLACEHOLDER_DOMAINS


def _normalize_key(name: Optional[str], email: str) -> Tuple[str, str]:
    """Stable dedup key. Names are lowercased + whitespace-trimmed;
    emails always lowercased so 'Dani@X.com' and 'dani@x.com' merge."""
    nm = (name or "").strip().lower()
    em = (email or "").strip().lower()
    return (nm, em)


def _bump(bucket: Dict[Tuple[str, str], Dict[str, Any]],
          name: Optional[str], email: str,
          source: str, ts: float,
          source_label: Optional[str] = None) -> None:
    """Merge a single (name, email, source) tuple into the bucket dict.
    Picks the highest confidence seen; tracks all sources and the
    most-recent timestamp."""
    if _is_placeholder_email(email):
        return
    key = _normalize_key(name, email)
    entry = bucket.get(key)
    label = source_label or source
    if entry is None:
        bucket[key] = {
            "names": {name} if name else set(),
            "email": email,
            "sources": {label},
            "count": 1,
            "last_seen": float(ts),
            "confidence": _CONFIDENCE.get(source, 0.5),
        }
        return
    if name:
        entry["names"].add(name)
    entry["sources"].add(label)
    entry["count"] += 1
    if ts > entry["last_seen"]:
        entry["last_seen"] = float(ts)
    new_conf = _CONFIDENCE.get(source, 0.5)
    if new_conf > entry["confidence"]:
        entry["confidence"] = new_conf


def _from_memory_facts(store, bucket: Dict[Tuple[str, str], Dict[str, Any]],
                       cap: int = _PER_SOURCE_CAP) -> int:
    """SOURCE 1: semantic facts with kind in ('person','contact').
    These rows are written by memory.extractor when iris learns a
    person→email mapping from a sent mail or lookup result."""
    n = 0
    try:
        rows: List[Any] = []
        for kind in ("person", "contact"):
            try:
                rows.extend(store.find_facts(kind=kind, limit=cap * 2))
            except Exception:
                continue
        for fact in rows[: cap * 2]:
            val = (getattr(fact, "value", "") or "").strip()
            key_name = (getattr(fact, "key", "") or "").strip()
            if not val or "@" not in val:
                continue
            # Provenance label uses source_kind when present so the UI
            # can disambiguate 'learned from user_said' vs 'planner_step'.
            sk = getattr(fact, "source_kind", None) or "legacy"
            label = f"memory.person[{sk}]"
            _bump(bucket, key_name or None, val, "memory.person",
                  float(getattr(fact, "ts", 0.0) or 0.0),
                  source_label=label)
            n += 1
            if n >= cap:
                break
    except Exception:
        # Soft fail — DB missing / migration mid-flight / etc.
        return n
    return n


def _from_episodic(store, bucket: Dict[Tuple[str, str], Dict[str, Any]],
                   cap: int = _PER_SOURCE_CAP) -> int:
    """SOURCE 2: scan episodic user_text for (proper-noun, email) pairs.
    Conservative: requires BOTH a capitalized word and an email in the
    same episode; rejects common English stopwords; caps total."""
    n = 0
    try:
        episodes = store.list_episodic(limit=500)
    except Exception:
        return 0
    for ep in episodes:
        if n >= cap:
            break
        text = (getattr(ep, "user_text", "") or "")
        if not text or "@" not in text:
            continue
        emails = {e for e in _EMAIL_RE.findall(text)
                  if not _is_placeholder_email(e)}
        if not emails:
            continue
        names = {nm for nm in _NAME_RE.findall(text)
                 if nm not in _NAME_STOPWORDS and len(nm) >= 2}
        ts = float(getattr(ep, "ts", 0.0) or 0.0)
        if not names:
            # Email-only mentions still count, but with no display name.
            for em in emails:
                _bump(bucket, None, em, "episodic.mention", ts)
                n += 1
                if n >= cap:
                    break
            continue
        # Pair every plausible name with every email in the same
        # episode. Cheap and good enough for short user turns.
        for em in emails:
            for nm in names:
                _bump(bucket, nm, em, "episodic.mention", ts)
                n += 1
                if n >= cap:
                    break
            if n >= cap:
                break
    return n


def _from_ms365_contacts(bucket: Dict[Tuple[str, str], Dict[str, Any]],
                         cap: int = _PER_SOURCE_CAP) -> int:
    """SOURCE 3: MS365 contacts_search across all connected accounts.
    Uses a broad wildcard query so we surface the address book rather
    than zero. Hard-capped per-call to avoid Graph throttling."""
    try:
        from hgr.live_api.connectors.ms365_connector import Microsoft365Connector
        from hgr.live_api.connectors.ms_graph_client import MsGraphClient
    except Exception:
        return 0
    n = 0
    try:
        client = MsGraphClient()
        if not getattr(client, "ready", lambda: False)():
            return 0
        conn = Microsoft365Connector(client)
        # Two passes with different seed letters — Graph contacts_search
        # always requires a non-empty query; 'a' + 'e' covers the bulk
        # of English/Latin names without over-fetching.
        seeds = ("a", "e")
        per_call = max(5, min(20, cap // 2))
        now_ts = time.time()
        for seed in seeds:
            if n >= cap:
                break
            try:
                result = conn.execute("contacts_search",
                                      {"query": seed, "max": per_call})
            except Exception:
                continue
            if not isinstance(result, dict) or result.get("status") != "ok":
                continue
            for contact in (result.get("contacts") or []):
                name = (contact.get("name") or "").strip() or None
                src = contact.get("source") or "?"
                acct = contact.get("source_account") or "?"
                label = f"ms365.contacts[{acct}.{src}]"
                for em in (contact.get("emails") or []):
                    em = (em or "").strip()
                    if not em:
                        continue
                    _bump(bucket, name, em, "ms365.contacts",
                          now_ts, source_label=label)
                    n += 1
                    if n >= cap:
                        break
                if n >= cap:
                    break
    except Exception:
        return n
    return n


def _from_gmail(bucket: Dict[Tuple[str, str], Dict[str, Any]],
                cap: int = _PER_SOURCE_CAP) -> int:
    """SOURCE 4: gmail_list senders (From: header)."""
    try:
        from hgr.live_api.connectors.gmail_connector import GmailConnector
    except Exception:
        return 0
    n = 0
    try:
        conn = GmailConnector()
        if not getattr(conn, "available", lambda: False)():
            return 0
        try:
            result = conn.execute("gmail_list",
                                  {"max": min(30, cap),
                                   "include_body": False})
        except Exception:
            return 0
        if not isinstance(result, dict) or result.get("status") != "ok":
            return 0
        now_ts = time.time()
        for msg in (result.get("messages") or []):
            if n >= cap:
                break
            addr = (msg.get("from") or "").strip()
            if not addr or "@" not in addr:
                continue
            nm = (msg.get("from_name") or "").strip() or None
            _bump(bucket, nm, addr, "gmail.sender", now_ts)
            n += 1
    except Exception:
        return n
    return n


def _from_ms_calendar(bucket: Dict[Tuple[str, str], Dict[str, Any]],
                      cap: int = _PER_SOURCE_CAP) -> int:
    """SOURCE 5: MS365 calendar attendees. The ms_calendar_list tool
    only returns subject/start/link, so we list event ids and expand a
    handful one-by-one to read the attendees array. Capped tight so a
    busy calendar doesn't burn the Graph quota."""
    try:
        from hgr.live_api.connectors.ms365_connector import Microsoft365Connector
        from hgr.live_api.connectors.ms_graph_client import MsGraphClient
    except Exception:
        return 0
    n = 0
    try:
        client = MsGraphClient()
        if not getattr(client, "ready", lambda: False)():
            return 0
        conn = Microsoft365Connector(client)
        # Direct Graph call: list events with attendees inline so we
        # don't need a second hop per event. _graph is the connector's
        # raw helper (already authenticated against the default account).
        try:
            data, err = conn._graph(  # type: ignore[attr-defined]
                "GET",
                "/me/events?$top=10&$orderby=start/dateTime DESC"
                "&$select=subject,start,attendees,organizer",
            )
        except Exception:
            return 0
        if err or not isinstance(data, dict):
            return 0
        now_ts = time.time()
        for ev in (data.get("value") or []):
            if n >= cap:
                break
            # Organizer counts as a contact too.
            org = ((ev.get("organizer") or {}).get("emailAddress") or {})
            org_addr = (org.get("address") or "").strip()
            org_name = (org.get("name") or "").strip() or None
            if org_addr and "@" in org_addr:
                _bump(bucket, org_name, org_addr, "ms365.calendar", now_ts)
                n += 1
            for att in (ev.get("attendees") or []):
                if n >= cap:
                    break
                ea = (att.get("emailAddress") or {})
                addr = (ea.get("address") or "").strip()
                nm = (ea.get("name") or "").strip() or None
                if not addr or "@" not in addr:
                    continue
                _bump(bucket, nm, addr, "ms365.calendar", now_ts)
                n += 1
    except Exception:
        return n
    return n


def _git_log_for_root(root: Path, max_commits: int = 50) -> List[Tuple[str, str, float]]:
    """Read recent git log entries from one project root. Returns
    [(name, email, ts), ...]. Empty on any failure (no .git, git not
    installed, permission, encoding)."""
    if not root or not root.exists():
        return []
    git_dir = root / ".git"
    if not git_dir.exists():
        return []
    try:
        # %x1f = unit separator, %x1e = record separator — robust against
        # names/emails containing commas/pipes.
        fmt = "%aN\x1f%aE\x1f%at"
        out = subprocess.run(
            ["git", "-C", str(root), "log",
             f"-n{int(max_commits)}", f"--format={fmt}"],
            capture_output=True, text=True, timeout=10,
            encoding="utf-8", errors="replace",
        )
        if out.returncode != 0:
            return []
    except Exception:
        return []
    rows: List[Tuple[str, str, float]] = []
    for line in (out.stdout or "").splitlines():
        parts = line.split("\x1f")
        if len(parts) < 3:
            continue
        nm = (parts[0] or "").strip()
        em = (parts[1] or "").strip()
        try:
            ts = float(parts[2])
        except (TypeError, ValueError):
            ts = 0.0
        if em and "@" in em:
            rows.append((nm or None, em, ts))
    return rows


def _from_git(projects: Optional[Iterable[Any]],
              bucket: Dict[Tuple[str, str], Dict[str, Any]],
              cap: int = _PER_SOURCE_CAP) -> int:
    """SOURCE 6: per-project git log authors. Iterates known project
    roots (when supplied) plus the host's global git config. Caps each
    project at 50 commits per the task spec."""
    n = 0

    # Resolve project roots from a few accepted shapes — callers pass
    # either a list of Path-like things or {project_id: ProjectMemoryStore}.
    roots: List[Path] = []
    if projects:
        try:
            if isinstance(projects, dict):
                iterable = projects.values()
            else:
                iterable = projects
            for entry in iterable:
                root = None
                if isinstance(entry, (str, os.PathLike)):
                    root = Path(entry)
                elif isinstance(entry, dict):
                    cand = entry.get("root") or entry.get("path")
                    if cand:
                        root = Path(cand)
                else:
                    # ProjectMemoryStore-ish duck typing — try .root then
                    # .project_root attributes.
                    cand = (getattr(entry, "root", None)
                            or getattr(entry, "project_root", None))
                    if cand:
                        root = Path(cand)
                if root is not None:
                    roots.append(root)
        except Exception:
            pass

    # If no projects passed, fall back to the hardcoded sibling roots.
    if not roots:
        try:
            from hgr.live_api.known_projects import KNOWN_PROJECT_ROOTS
            roots = [Path(p["root"]) for p in KNOWN_PROJECT_ROOTS]
        except Exception:
            roots = []

    seen_roots: set = set()
    for root in roots:
        if n >= cap:
            break
        try:
            key = str(root.resolve()).lower()
        except Exception:
            key = str(root).lower()
        if key in seen_roots:
            continue
        seen_roots.add(key)
        for nm, em, ts in _git_log_for_root(root, max_commits=50):
            _bump(bucket, nm, em, "git.config",
                  ts or time.time(),
                  source_label=f"git.config[{root.name}]")
            n += 1
            if n >= cap:
                break

    # Always also try the global git identity — picks up the current
    # dev even when no project root has commits attributed to them yet.
    if n < cap:
        try:
            name_p = subprocess.run(
                ["git", "config", "--global", "user.name"],
                capture_output=True, text=True, timeout=5,
                encoding="utf-8", errors="replace",
            )
            email_p = subprocess.run(
                ["git", "config", "--global", "user.email"],
                capture_output=True, text=True, timeout=5,
                encoding="utf-8", errors="replace",
            )
            nm = (name_p.stdout or "").strip()
            em = (email_p.stdout or "").strip()
            if em and "@" in em:
                _bump(bucket, nm or None, em, "git.config",
                      time.time(), source_label="git.config[global]")
                n += 1
        except Exception:
            pass
    return n


def gather_contacts(memory_store: Any,
                    projects: Optional[Iterable[Any]] = None,
                    max_contacts: int = 80) -> List[Dict[str, Any]]:
    """Aggregate contacts from every available source and return a
    deduplicated, ranked list.

    Args:
        memory_store: A ``MemoryStore`` (SQLite-backed). Pass ``None``
            to skip the two memory-backed sources.
        projects: Iterable of project entries — either Path-likes, dicts
            with a 'root'/'path' key, or ProjectMemoryStore-style
            objects with a ``root`` attribute. When omitted, falls back
            to ``known_projects.KNOWN_PROJECT_ROOTS``.
        max_contacts: Hard cap on returned contacts. Default 80.

    Returns:
        List of dicts shaped as
        ``{id, name, email, sources: [...], count, last_seen,
        confidence}``. Sorted by confidence desc, then last_seen desc,
        then count desc. ``last_seen`` is an ISO-8601 string. ``id`` is
        a short uuid hex slug suitable for use as a stable DOM id.

    Never raises — every source is independently try/excepted, and on
    a total wipeout we return ``[]``.
    """
    bucket: Dict[Tuple[str, str], Dict[str, Any]] = {}

    if memory_store is not None:
        try:
            _from_memory_facts(memory_store, bucket)
        except Exception:
            pass
        try:
            _from_episodic(memory_store, bucket)
        except Exception:
            pass

    # HTTP-backed sources — bounded by a hard timeout each so the
    # gatherer can never stall the simulator launch on a slow connector.
    # ``_run_source_with_timeout`` swallows any exception/timeout and
    # returns 0 so the rest of the pipeline keeps going.
    _run_source_with_timeout(
        lambda: _from_ms365_contacts(bucket), "ms365_contacts",
    )
    _run_source_with_timeout(
        lambda: _from_gmail(bucket), "gmail",
    )
    _run_source_with_timeout(
        lambda: _from_ms_calendar(bucket), "ms_calendar",
    )

    # Git log is a local subprocess call — already has per-command
    # timeouts in _git_log_for_root, no need to wrap it again.
    try:
        _from_git(projects, bucket)
    except Exception:
        pass

    # Finalize: pick a preferred name (longest non-empty), normalize the
    # sources list, format the timestamp, drop the in-memory sets.
    out: List[Dict[str, Any]] = []
    for (name_key, email_key), data in bucket.items():
        names = [n for n in data["names"] if n]
        preferred = (sorted(names, key=len, reverse=True)[0]
                     if names else email_key.split("@", 1)[0])
        try:
            iso_ts = datetime.fromtimestamp(
                float(data["last_seen"] or time.time())
            ).isoformat()
        except (OverflowError, OSError, ValueError):
            iso_ts = datetime.now().isoformat()
        out.append({
            "id": uuid.uuid4().hex[:8],
            "name": preferred,
            "email": data["email"],
            "sources": sorted(data["sources"]),
            "count": int(data["count"]),
            "last_seen": iso_ts,
            "confidence": float(data["confidence"]),
        })

    # Classify each contact as 'person' or 'transactional' before any
    # truncation — caller can split or filter as needed.
    for c in out:
        c["kind"] = classify_contact(c)

    # Rank: confidence first, then recency, then frequency.
    def _sort_key(c: Dict[str, Any]) -> Tuple[float, float, int]:
        try:
            recency = datetime.fromisoformat(c["last_seen"]).timestamp()
        except Exception:
            recency = 0.0
        return (-float(c["confidence"]), -recency, -int(c["count"]))

    out.sort(key=_sort_key)
    if max_contacts and len(out) > max_contacts:
        out = out[: int(max_contacts)]
    return out


# ── transactional / marketing classification ──────────────────────────
# A contact is a real PERSON if any of its sources are person-verified
# (memory.person, ms365.contacts, episodic.mention, git.config). It is
# TRANSACTIONAL if it only comes from gmail.sender AND looks like a
# brand / newsletter / automated-system address.
_PERSON_VERIFIED_SOURCE_PREFIXES = (
    "memory.person",
    "ms365.contacts",
    "ms365.calendar",
    "git.config",
)

# Email LOCAL-PARTS (before @) that scream automated:
_TRANSACTIONAL_LOCAL_PARTS = (
    "noreply", "no-reply", "no.reply", "donotreply", "do-not-reply", "do.not.reply",
    "notifications", "notification", "notify",
    "newsletter", "newsletters", "news",
    "marketing", "promotions", "promo", "offers",
    "alerts", "alert", "digest", "updates", "update",
    "support", "help", "info", "hello", "contact",
    "team", "automated", "system", "bot", "mailer",
    "billing", "invoice", "receipt", "orders", "order",
    "feedback", "service", "services", "admin",
    "auto-confirm", "auto-reply", "transactional",
    "account", "accounts", "security",
)

# Substrings that, when found in the DISPLAY NAME, mark transactional.
# Real people rarely have any of these in their name. Word-boundary
# matched against the lowercased name in classify_contact below.
_TRANSACTIONAL_NAME_SUBSTRINGS = (
    # Automated-sender words
    "team", "support", "notifications", "notification", "alert", "alerts",
    "updates", "update", "newsletter", "news",
    "noreply", "no-reply", "do not reply",
    "automated", "auto-reply", "auto-confirm",
    "feedback", "service", "services",
    # Marketing / commerce words
    "bundle", "deals", "offers", "deal", "marketing", "promotion", "promo",
    "digest", "weekly", "daily", "billing", "invoice", "receipt",
    "order", "orders", "subscription", "subscriptions",
    "rewards", "loyalty", "membership",
    # Brand/product/service indicators
    "card", "bank", "credit", "loan", "mortgage",
    "store", "shoppe", "shop", "market", "warehouse",
    "center", "centre", "club", "academy", "institute",
    "premium", "premier", "pro", "plus", "elite",
    "developer", "developers", "acrobat", "adobe",
    "studio", "studios", "labs", "media", "press",
    "inc", "llc", "ltd", "co.", "corp", "company",
    # Job/recruitment platforms (all brands)
    "linkedin", "indeed", "glassdoor", "monster", "ziprecruiter",
    "linkedin job", "job alerts", "jobs",
)

# Display-name prefixes that indicate a brand/org, not a person.
_TRANSACTIONAL_NAME_PREFIXES = ("the ",)

# Personal email domains — real people typically use these. A name+email
# combo where the domain is NOT one of these AND the local-part doesn't
# look name-shaped is treated as transactional.
_PERSONAL_EMAIL_DOMAINS = (
    "gmail.com", "googlemail.com",
    "yahoo.com", "ymail.com", "rocketmail.com",
    "outlook.com", "hotmail.com", "live.com", "msn.com",
    "icloud.com", "me.com", "mac.com",
    "aol.com", "protonmail.com", "proton.me",
    "fastmail.com", "tutanota.com", "hey.com",
    "mail.com", "zoho.com",
)


def classify_contact(contact: Dict[str, Any]) -> str:
    """Return 'person' or 'transactional' for a single gathered contact.

    Rule of thumb: anything with a person-verified source (memory facts,
    ms365 address book, calendar attendees, git commits) is a person.
    Anything that ONLY appears as a gmail sender is judged by its name
    and email. Brand names and known automated patterns -> transactional.
    """
    sources = contact.get("sources") or []
    src_lower = [str(s).lower() for s in sources]
    has_verified = any(
        any(s.startswith(prefix) for prefix in _PERSON_VERIFIED_SOURCE_PREFIXES)
        for s in src_lower
    )
    if has_verified:
        return "person"

    email = (contact.get("email") or "").lower()
    name_raw = (contact.get("name") or "")
    name = name_raw.lower()

    # 1. Email local-part check (orders@..., noreply@..., etc.)
    if "@" in email:
        local, _, domain = email.partition("@")
        for part in _TRANSACTIONAL_LOCAL_PARTS:
            if part in local:
                return "transactional"
        # 2. Domain check: non-personal-domain + non-name-shaped local
        #    => transactional. Catches orders@apple.com, info@brand.com.
        if domain and not any(domain.endswith(d) for d in _PERSONAL_EMAIL_DOMAINS):
            # Local-part looking like a name: contains a dot OR matches
            # firstname/firstinitial+lastname pattern. Otherwise treat
            # as transactional.
            has_dot = "." in local
            looks_name_like = has_dot and not any(ch.isdigit() for ch in local)
            if not looks_name_like:
                return "transactional"

    # 3. Name prefix check ("The X" => brand/org)
    for prefix in _TRANSACTIONAL_NAME_PREFIXES:
        if name.startswith(prefix):
            return "transactional"

    # 4. Name substring check — word-boundary so "Card" doesn't fire
    #    inside "Cardinal". Match against whitespace-split tokens.
    name_tokens = set(t.lower().strip(".,;:!?") for t in name_raw.split())
    for sub in _TRANSACTIONAL_NAME_SUBSTRINGS:
        if " " in sub:
            if sub in name:
                return "transactional"
        elif sub in name_tokens:
            return "transactional"

    # 5. Display-name heuristic: single-token names are usually brands
    #    (LinkedIn, Spotify, H&M). Real people have at least two tokens.
    tokens = [t for t in name_raw.split() if t]
    if len(tokens) < 2:
        return "transactional"
    # All-caps short names (AMAZON ORDER, ADOBE PRO) => transactional.
    if name_raw.isupper() and len(tokens) <= 3:
        return "transactional"

    return "person"


# ── relevance-scoring enrichment ──────────────────────────────────────
# Day-of-week labels in Python's weekday() order (Mon=0 .. Sun=6).
_DOW_LABELS = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")

# Time window (seconds) around each episode.ts used when joining the
# external tool_call_log.db. Episodes don't carry session_id, so we
# fall back to "tools fired within ±N seconds" as a proxy for
# "same conversation turn". 30 min is loose enough to cover slow
# multi-step planner runs without leaking across unrelated sessions.
_TOOL_TIME_WINDOW_S = 1800.0

# Minimum length of a contact name for relevance scanning. Skips
# things like "Ai" or "a" that would explode false positives. Note
# the lowercase comparison done in enrich_contacts_with_relevance.
_MIN_NAME_LEN_FOR_RELEVANCE = 3


def _build_project_alias_map(
    projects: Optional[Iterable[Any]],
) -> Dict[str, str]:
    """Build a {lowercase_keyword -> project_id} map for project
    inference. Accepts the same project shapes as :func:`_from_git`
    (dicts with 'id'/'label', Path-likes, ProjectMemoryStore-ish
    objects). Falls back to ``KNOWN_PROJECT_ROOTS`` when empty.

    Tokens added per project: its id, its label (lowercased), and the
    individual words inside the label longer than 3 chars (so
    "Touchless website" matches "touchless" or "website").
    """
    alias_to_pid: Dict[str, str] = {}

    def _add_tokens(pid: str, label: str) -> None:
        pid_l = (pid or "").strip().lower()
        if pid_l:
            alias_to_pid.setdefault(pid_l, pid)
        lbl = (label or "").strip().lower()
        if lbl:
            alias_to_pid.setdefault(lbl, pid)
            for tok in re.split(r"[\s\-_/]+", lbl):
                tok = tok.strip()
                if len(tok) > 3:
                    alias_to_pid.setdefault(tok, pid)

    entries: List[Any] = []
    if projects:
        try:
            if isinstance(projects, dict):
                entries = list(projects.values())
            else:
                entries = list(projects)
        except Exception:
            entries = []
    if not entries:
        try:
            from hgr.live_api.known_projects import KNOWN_PROJECT_ROOTS
            entries = list(KNOWN_PROJECT_ROOTS)
        except Exception:
            entries = []

    for entry in entries:
        try:
            if isinstance(entry, dict):
                pid = str(entry.get("id") or "").strip()
                label = str(entry.get("label") or pid or "").strip()
                _add_tokens(pid, label)
            elif isinstance(entry, (str, os.PathLike)):
                root = Path(entry)
                pid = root.name.lower()
                _add_tokens(pid, pid)
            else:
                pid = (getattr(entry, "id", None)
                       or getattr(entry, "project_id", None) or "")
                label = (getattr(entry, "label", None) or pid or "")
                _add_tokens(str(pid), str(label))
        except Exception:
            continue
    return alias_to_pid


def _extract_tools_from_steps_json(raw: Optional[str]) -> List[str]:
    """Pull tool ids out of a stored ``steps_json`` blob. Defensive
    against any shape — accepts ``[{"tool": "x"}, ...]`` (current
    schema, see manager._steps_to_json) and a few legacy variants."""
    if not raw:
        return []
    try:
        data = json.loads(raw)
    except (TypeError, ValueError):
        return []
    if not isinstance(data, list):
        return []
    out: List[str] = []
    for row in data:
        if isinstance(row, dict):
            tid = row.get("tool") or row.get("tool_id") or row.get("name")
            if tid:
                out.append(str(tid))
        elif isinstance(row, str):
            out.append(row)
    return out


def _index_tool_log_by_window(
    tool_log_path: Optional[Path],
) -> List[Tuple[float, str]]:
    """Load ``tool_call_log.db`` rows as a sorted list of
    ``(ts, tool_id)`` tuples for cheap bisect-style window lookups.

    Returns ``[]`` on any failure (missing file, missing module, broken
    DB). A contact still gets steps_json-derived tools when this fails.
    """
    if not tool_log_path:
        return []
    try:
        path = Path(tool_log_path)
    except Exception:
        return []
    if not path.exists():
        return []
    try:
        from hgr.live_api.cortex.tool_call_log import load_sessions
    except Exception:
        return []
    try:
        sessions = load_sessions(path)
    except Exception:
        return []
    # load_sessions doesn't carry per-tool ts — so derive a coarse ts
    # from each session's earliest_ts. Good enough for ±30 min joins.
    rows: List[Tuple[float, str]] = []
    for sess in sessions:
        sts = float(sess.get("earliest_ts") or 0.0)
        for tid in (sess.get("tool_ids") or []):
            if tid:
                rows.append((sts, str(tid)))
    rows.sort(key=lambda r: r[0])
    return rows


def _tools_in_window(
    indexed: List[Tuple[float, str]],
    center_ts: float,
    window_s: float = _TOOL_TIME_WINDOW_S,
) -> List[str]:
    """Linear scan — cheap given typical N (hundreds of tool rows)."""
    if not indexed or not center_ts:
        return []
    lo = center_ts - window_s
    hi = center_ts + window_s
    return [tid for ts, tid in indexed if lo <= ts <= hi]


def enrich_contacts_with_relevance(
    contacts: List[Dict[str, Any]],
    memory_store: Any,
    tool_call_log_path: Optional[Path] = None,
    max_episodes: int = 500,
    projects: Optional[Iterable[Any]] = None,
) -> List[Dict[str, Any]]:
    """Decorate each contact in-place with relevance fields mined from
    the episodic memory + tool_call_log.

    Adds these keys per contact (omits keys the source can't supply):

    * ``typical_hours``    — up to 3 most-common hours-of-day (0-23) the
                             contact was mentioned in user_text.
    * ``typical_days``     — up to 3 most-common weekday names
                             (``Mon``..``Sun``) of those mentions.
    * ``related_tools``    — up to 5 tool_ids that fired alongside the
                             contact (from the episode's own
                             ``steps_json`` + best-effort time-window
                             join with ``tool_call_log.db``).
    * ``related_projects`` — up to 3 project_ids whose name/label
                             appears in the episode text or
                             ``plan_json.goal`` during the same mention.
    * ``mention_count``    — total number of episodes that matched.

    Matching is **case-insensitive whole-word** (``\\b``-bounded). Names
    shorter than 3 characters are skipped — "Ai" or "Al" would otherwise
    fire on almost every episode. Email is matched as a literal
    substring (already unique enough).

    Each contact's enrichment is wrapped in its own try/except so one
    bad row never derails the rest. The original ``contacts`` list is
    returned with the new keys grafted on; existing keys are untouched.
    """
    if not contacts or memory_store is None:
        return contacts

    # Pull episodic rows once. Most recent first. Bounded by max_episodes
    # so a 50k-row DB doesn't kill the launcher.
    try:
        episodes = list(memory_store.list_episodic(limit=int(max_episodes)))
    except Exception:
        return contacts

    # Pre-extract per-episode signals so we don't re-parse for every
    # contact. Each row carries: text_lower, ts, hour, dow, tools (from
    # steps_json), goal_text.
    pre: List[Dict[str, Any]] = []
    for ep in episodes:
        try:
            txt = (getattr(ep, "user_text", "") or "")
            txt_l = txt.lower()
            ts = float(getattr(ep, "ts", 0.0) or 0.0)
            try:
                dt = datetime.fromtimestamp(ts) if ts else None
                hour = dt.hour if dt else None
                dow = _DOW_LABELS[dt.weekday()] if dt else None
            except (ValueError, OverflowError, OSError):
                hour, dow = None, None
            ep_tools = _extract_tools_from_steps_json(
                getattr(ep, "steps_json", None)
            )
            goal_text = ""
            try:
                plan_obj = json.loads(getattr(ep, "plan_json", None) or "{}")
                if isinstance(plan_obj, dict):
                    goal_text = str(plan_obj.get("goal") or "").lower()
            except (TypeError, ValueError):
                goal_text = ""
            pre.append({
                "text_l": txt_l,
                "ts": ts,
                "hour": hour,
                "dow": dow,
                "tools": ep_tools,
                "goal_l": goal_text,
            })
        except Exception:
            continue

    # Best-effort tool_call_log index for time-window joins. Empty when
    # the DB doesn't exist or the module fails to import — that's fine,
    # we still surface steps_json-derived tools.
    tool_index = _index_tool_log_by_window(tool_call_log_path)

    # Project alias map for related_projects inference.
    alias_to_pid = _build_project_alias_map(projects)

    for contact in contacts:
        try:
            name_raw = str(contact.get("name") or "").strip()
            email_raw = str(contact.get("email") or "").strip().lower()
            name_l = name_raw.lower()

            # Skip enrichment when name is too short to safely match.
            # Email substring search is still safe though — fall back to
            # email-only if available.
            name_searchable = name_l if len(name_l) >= _MIN_NAME_LEN_FOR_RELEVANCE else ""
            email_searchable = email_raw if "@" in email_raw else ""

            if not name_searchable and not email_searchable:
                # Nothing to match against — leave contact unchanged.
                continue

            # Pre-compile a \b-bounded regex for the name. Escape so
            # special chars (e.g. "O'Brien", "Jean-Luc") don't break it.
            name_re = None
            if name_searchable:
                try:
                    name_re = re.compile(
                        r"\b" + re.escape(name_searchable) + r"\b"
                    )
                except re.error:
                    name_re = None

            hours: Dict[int, int] = {}
            days: Dict[str, int] = {}
            tools: Dict[str, int] = {}
            projects_hit: Dict[str, int] = {}
            mention_count = 0

            for row in pre:
                text_l = row["text_l"]
                if not text_l:
                    continue
                matched = False
                if name_re is not None and name_re.search(text_l):
                    matched = True
                elif email_searchable and email_searchable in text_l:
                    matched = True
                if not matched:
                    continue

                mention_count += 1

                # Hour / day buckets.
                if row["hour"] is not None:
                    hours[row["hour"]] = hours.get(row["hour"], 0) + 1
                if row["dow"]:
                    days[row["dow"]] = days.get(row["dow"], 0) + 1

                # Tools from the episode's own steps_json — strongest
                # "same session" signal we have.
                for tid in row["tools"]:
                    tools[tid] = tools.get(tid, 0) + 1

                # Tools from tool_call_log (time-window join), folded
                # in with lower weight (de-duped by id, count once per
                # episode regardless of how many fired in the window).
                if tool_index:
                    win_tools = set(_tools_in_window(tool_index, row["ts"]))
                    for tid in win_tools:
                        tools[tid] = tools.get(tid, 0) + 1

                # Project inference: scan goal_text + user_text for
                # alias keywords. Each alias hit counts once per episode
                # to avoid one chatty goal sentence dominating the rank.
                if alias_to_pid:
                    seen_proj_this_ep: set = set()
                    haystack = row["goal_l"] + " " + text_l
                    for alias, pid in alias_to_pid.items():
                        if pid in seen_proj_this_ep:
                            continue
                        # Word-boundary match so "ai" alias doesn't
                        # explode in "email" / "mail".
                        if re.search(
                            r"\b" + re.escape(alias) + r"\b",
                            haystack,
                        ):
                            projects_hit[pid] = projects_hit.get(pid, 0) + 1
                            seen_proj_this_ep.add(pid)

            # Skip contacts with zero matches — don't pollute with empty
            # fields; downstream UI can use ``.get(..., [])``.
            if mention_count == 0:
                continue

            typical_hours = [
                h for h, _ in sorted(
                    hours.items(), key=lambda kv: (-kv[1], kv[0])
                )[:3]
            ]
            typical_days = [
                d for d, _ in sorted(
                    days.items(), key=lambda kv: (-kv[1], _DOW_LABELS.index(kv[0]))
                )[:3]
            ]
            related_tools = [
                t for t, _ in sorted(
                    tools.items(), key=lambda kv: (-kv[1], kv[0])
                )[:5]
            ]
            related_projects = [
                p for p, _ in sorted(
                    projects_hit.items(), key=lambda kv: (-kv[1], kv[0])
                )[:3]
            ]

            contact["typical_hours"] = typical_hours
            contact["typical_days"] = typical_days
            contact["related_tools"] = related_tools
            contact["related_projects"] = related_projects
            contact["mention_count"] = int(mention_count)
        except Exception:
            # One bad contact must not break the rest. Leave it untouched.
            continue

    return contacts


__all__ = [
    "gather_contacts",
    "classify_contact",
    "enrich_contacts_with_relevance",
]
