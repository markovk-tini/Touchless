"""web_search — structured web search built-in.

Returns {title, url, snippet} per result instead of a screenshot or raw
HTML, so the planner can pick a result and `web_navigate` straight to it
without OCR or DOM scraping. Two providers:

  1. Google Custom Search API (preferred) — needs GOOGLE_CSE_API_KEY +
     GOOGLE_CSE_ID env vars. Free tier: 100 queries/day. Clean structured
     results, very reliable.
  2. DuckDuckGo HTML (fallback) — no API key needed. Less reliable because
     it parses HTML, but works out of the box. Used only when CSE isn't
     configured.

Returns a uniform dict shape:
  {"status": "ok", "provider": "<name>",
   "results": [{"title": ..., "url": ..., "snippet": ...}, ...]}
or on failure:
  {"status": "error", "error": "<msg>", "code": "<code>"}

Author: Konstantin Markov
"""
from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, List

_GOOGLE_CSE_URL = "https://www.googleapis.com/customsearch/v1"
_DDG_HTML_URL = "https://html.duckduckgo.com/html/"
_TIMEOUT = 8.0
_DEFAULT_COUNT = 5
_MAX_COUNT = 10
_MAX_QUERY_CHARS = 500   # CSE accepts ~2K; cap so we never send arbitrary text.
_MAX_RECENT_DAYS = 365

# Path tokens that strongly suggest a category/section landing page rather
# than a specific article. Used to demote those results so the planner's
# default {step:N.results[0].url} picks a real article.
_SECTION_TOKENS = frozenset({
    "category", "categories", "section", "sections", "topic", "topics",
    "tag", "tags", "subject", "subjects", "channel", "channels",
    "hub", "feed", "feeds", "archive", "archives", "index",
    "browse", "all", "latest",
})


def _article_score(url: str) -> int:
    """Heuristic: how 'article-like' is this URL? Higher = more likely to be
    a specific story rather than a category landing page.

    Cues:
      +3  A YYYY/MM/DD date segment in the path (very strong article signal).
      +2  A long slug at the end (>30 chars, dashes/underscores — typical
          article URL shape: '/2026/05/anthropic-launches-next-gen-model').
      -3  A path segment is a known section/category token ('category',
          'topic', 'tag', etc.).
      -2  Path has 2 or fewer segments AND the last is short with no
          dashes (e.g. '/technology/' or '/ai' — typical category roots).
    """
    try:
        path = urllib.parse.urlparse(url).path or ""
    except Exception:
        return 0
    parts = [p for p in path.split("/") if p]
    score = 0
    # Date in path = strong article signal.
    if any(
        len(p) == 4 and p.isdigit() and 1900 <= int(p) <= 2100
        for p in parts
    ):
        # And the next segment is a month-shaped number.
        for i, p in enumerate(parts):
            if (len(p) == 4 and p.isdigit() and i + 1 < len(parts)
                    and parts[i + 1].isdigit()
                    and 1 <= int(parts[i + 1]) <= 12):
                score += 3
                break
    # Long slug at the end (the article title encoded as a URL slug).
    if parts:
        last = parts[-1].split("?")[0].split("#")[0]
        if len(last) > 30 and ("-" in last or "_" in last):
            score += 2
    # Section/category tokens demote.
    if any(p.lower() in _SECTION_TOKENS for p in parts):
        score -= 3
    # Short, dash-less last segment with a shallow path = likely a hub.
    if 0 < len(parts) <= 2:
        last = parts[-1]
        if len(last) < 25 and "-" not in last and "_" not in last:
            score -= 2
    return score


def web_search(query: str, count: int = _DEFAULT_COUNT,
               site: str = "", recent_days: int = 0) -> Dict[str, Any]:
    """Search the web. `site` restricts to a domain ("nytimes.com"),
    `recent_days` biases toward recent results when the provider supports
    it (Google CSE: 'dateRestrict=d7' etc.)."""
    q = (query or "").strip()
    if not q:
        return {"status": "error", "error": "empty query", "code": "invalid_arguments"}
    # Sanitize: cap the query so a runaway prompt doesn't end up in a URL,
    # and clamp recent_days so we never produce malformed 'dateRestrict=d-7'.
    q = q[:_MAX_QUERY_CHARS]
    count = max(1, min(int(count or _DEFAULT_COUNT), _MAX_COUNT))
    try:
        recent_days = max(0, min(int(recent_days or 0), _MAX_RECENT_DAYS))
    except (TypeError, ValueError):
        recent_days = 0
    if site:
        q = f"{q} site:{site}"

    # Google CSE first when configured.
    api_key = (os.environ.get("GOOGLE_CSE_API_KEY") or "").strip()
    cse_id = (os.environ.get("GOOGLE_CSE_ID") or "").strip()
    if api_key and cse_id:
        try:
            return _search_google_cse(q, api_key, cse_id, count, recent_days)
        except urllib.error.HTTPError as exc:
            # Quota exhausted / bad key → fall through to DDG; don't fail
            # the whole tool just because the user hit 100/day.
            if exc.code in (403, 429):
                pass
            else:
                return {"status": "error", "code": "cse_http_error",
                        "error": f"google CSE HTTP {exc.code}"}
        except Exception as exc:
            return {"status": "error", "code": "cse_failed",
                    "error": f"{type(exc).__name__}: {exc}"}

    # Fallback: DuckDuckGo HTML.
    try:
        return _search_duckduckgo(q, count)
    except Exception as exc:
        return {"status": "error", "code": "ddg_failed",
                "error": f"{type(exc).__name__}: {exc}"}


# ---- Google Custom Search ------------------------------------------------
def _search_google_cse(query: str, api_key: str, cse_id: str,
                       count: int, recent_days: int) -> Dict[str, Any]:
    params = {"key": api_key, "cx": cse_id, "q": query, "num": str(count)}
    if recent_days > 0:
        params["dateRestrict"] = f"d{int(recent_days)}"
    url = _GOOGLE_CSE_URL + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, method="GET",
                                 headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    results: List[Dict[str, str]] = []
    for item in (payload.get("items") or [])[:count]:
        results.append({
            "title": str(item.get("title") or "")[:200],
            "url": str(item.get("link") or ""),
            "snippet": str(item.get("snippet") or "")[:400],
        })
    return {"status": "ok", "provider": "google_cse",
            "results": _rank_articles_first(results)}


# ---- DuckDuckGo HTML fallback --------------------------------------------
# Each result is a block like:
#   <a class="result__a" href="REAL_URL_OR_REDIRECT">TITLE</a>
#   <a class="result__snippet" ...>SNIPPET</a>
# (DDG occasionally wraps href in a /l/?uddg= redirect.)
_RESULT_BLOCK_RE = re.compile(
    r'<a[^>]*class="[^"]*result__a[^"]*"[^>]*href="([^"]+)"[^>]*>(.*?)</a>'
    r'(?:.*?<a[^>]*class="[^"]*result__snippet[^"]*"[^>]*>(.*?)</a>)?',
    re.IGNORECASE | re.DOTALL,
)
_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")


def _strip_html(s: str) -> str:
    return _WS_RE.sub(" ", _TAG_RE.sub("", s or "")).strip()


def _unwrap_ddg_redirect(href: str) -> str:
    """DDG HTML sometimes returns '/l/?uddg=ENCODED'. Decode to the real URL."""
    if href.startswith("//duckduckgo.com/l/") or href.startswith("/l/"):
        try:
            q = urllib.parse.parse_qs(urllib.parse.urlparse(href).query)
            real = (q.get("uddg") or [""])[0]
            if real:
                return urllib.parse.unquote(real)
        except Exception:
            pass
    if href.startswith("//"):
        return "https:" + href
    return href


def _search_duckduckgo(query: str, count: int) -> Dict[str, Any]:
    data = urllib.parse.urlencode({"q": query, "kl": "us-en"}).encode("utf-8")
    req = urllib.request.Request(
        _DDG_HTML_URL, data=data, method="POST",
        headers={
            # DDG HTML endpoint rejects requests without a real-looking UA.
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                          "AppleWebKit/537.36 (KHTML, like Gecko) "
                          "Chrome/120.0 Safari/537.36",
            "Accept": "text/html",
            "Content-Type": "application/x-www-form-urlencoded",
        },
    )
    with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
        html = resp.read().decode("utf-8", errors="ignore")

    results: List[Dict[str, str]] = []
    for m in _RESULT_BLOCK_RE.finditer(html):
        href, title_html, snippet_html = m.group(1), m.group(2), m.group(3) or ""
        url = _unwrap_ddg_redirect(href)
        if not url.startswith("http"):
            continue
        results.append({
            "title": _strip_html(title_html)[:200],
            "url": url,
            "snippet": _strip_html(snippet_html)[:400],
        })
        if len(results) >= count:
            break
    return {"status": "ok", "provider": "duckduckgo",
            "results": _rank_articles_first(results)}


def _rank_articles_first(results: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Stable-sort results so article-shaped URLs land at the top, keeping
    the search engine's original ordering as the tiebreaker. The planner's
    default {step:N.results[0].url} pick then lands on a real article
    instead of a category landing page."""
    scored = [(idx, _article_score(r.get("url", "")), r)
              for idx, r in enumerate(results)]
    # Sort by score desc, then by original index asc (Python sorts are stable).
    scored.sort(key=lambda t: (-t[1], t[0]))
    return [r for _idx, _s, r in scored]
