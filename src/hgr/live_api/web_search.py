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
    return {"status": "ok", "provider": "google_cse", "results": results}


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
    return {"status": "ok", "provider": "duckduckgo", "results": results}
