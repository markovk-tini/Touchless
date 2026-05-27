"""Chrome DevTools Protocol (CDP) web controller for the Live API agent.

Lets Iris operate web pages by reading the REAL page (links, text,
elements) and navigating/clicking precisely — instead of guessing pixel
coordinates from a downscaled screenshot. This is both far more reliable
for web tasks and far cheaper token-wise (structured text, no images).

How it works:
  * Launches a DEDICATED Chrome instance with --remote-debugging-port and
    its own --user-data-dir (so it doesn't fight the user's everyday
    Chrome, which can't have remote debugging enabled after the fact).
  * Talks CDP over HTTP (target discovery) + WebSocket (commands), using
    the `requests` and `websocket-client` packages already vendored for
    the realtime client. No new dependency.

Exposed to the model via the web_* tools in schemas.py / tool_executor.py.
Synchronous and lock-guarded — tool calls run one at a time on the
websocket reader thread.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import parse_qs, urlparse


def _norm(s: str) -> str:
    """Lowercase + collapse runs of non-alphanumerics to single spaces, so
    matching ignores punctuation ("Valentine's" == "valentines")."""
    return re.sub(r"[^a-z0-9]+", " ", str(s).lower()).strip()


# Hosts that are the search engine's own UI/utility — never real results.
_ENGINE_CHROME_HOSTS = (
    "google.com", "google.co", "gstatic.com", "googleusercontent.com",
    "googleapis.com", "schema.org", "bing.com", "microsoft.com",
    "duckduckgo.com", "brave.com",
)


def _unwrap_redirect(href: str) -> str:
    """Google often wraps result links as /url?q=REAL. Return the real URL."""
    try:
        p = urlparse(href)
        if "google." in (p.netloc or "") and p.path.startswith("/url"):
            qs = parse_qs(p.query)
            real = (qs.get("q") or qs.get("url") or [""])[0]
            if real.startswith("http"):
                return real
    except Exception:
        pass
    return href


def _is_engine_chrome(href: str) -> bool:
    """True if href points to the search engine's own UI (nav, Maps, account)."""
    try:
        host = (urlparse(href).netloc or "").lower()
    except Exception:
        return False
    return any(host == h or host.endswith("." + h) for h in _ENGINE_CHROME_HOSTS)


def _resolve_url(target: str) -> str:
    """A bare query becomes a Google search; anything URL-ish is used as-is."""
    target = (target or "").strip()
    if not target:
        return "about:blank"
    if "://" in target or target.startswith(("http", "www.")):
        if target.startswith("www."):
            return "https://" + target
        return target
    if "." in target and " " not in target:
        return "https://" + target
    return "https://www.google.com/search?q=" + target.replace(" ", "+")


class WebController:
    def __init__(self, config, logger) -> None:
        self._config = config
        self._logger = logger
        self._port = int(os.environ.get("LIVE_API_CDP_PORT", "9222") or "9222")
        self._proc: Optional[subprocess.Popen] = None
        self._ws = None  # CDP websocket bound to the active page target
        self._msg_id = 0
        self._lock = threading.RLock()
        self._profile_dir = Path(config.safe_workspace_dir).parent / "chrome_cdp_profile"

    # ---- chrome lifecycle ----

    def _chrome_exe(self) -> Optional[str]:
        program_files = Path.home().anchor + "Program Files"
        program_files_x86 = Path.home().anchor + "Program Files (x86)"
        local = Path.home() / "AppData" / "Local"
        for p in (
            Path(program_files) / "Google" / "Chrome" / "Application" / "chrome.exe",
            Path(program_files_x86) / "Google" / "Chrome" / "Application" / "chrome.exe",
            local / "Google" / "Chrome" / "Application" / "chrome.exe",
        ):
            if p.exists():
                return str(p)
        return None

    def _cdp_http(self, path: str, timeout: float = 2.0):
        import requests  # local import — keeps module importable without it
        # Use "localhost", not 127.0.0.1: Chrome's DevTools endpoint rejects
        # WebSocket handshakes whose Host header is a bare IP (403 Forbidden,
        # a DNS-rebinding guard added ~Chrome 111). The webSocketDebuggerUrl
        # mirrors the host we query with, so querying via localhost yields a
        # localhost ws:// URL that the handshake accepts.
        url = f"http://localhost:{self._port}{path}"
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
        return resp.json()

    def _launch_chrome(self) -> bool:
        exe = self._chrome_exe()
        if not exe:
            self._logger.event("web_chrome_not_found")
            return False
        try:
            self._profile_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        args = [
            exe,
            f"--remote-debugging-port={self._port}",
            f"--user-data-dir={self._profile_dir}",
            # Allow the CDP WebSocket handshake (newer Chrome blocks it
            # unless the origin is explicitly permitted).
            "--remote-allow-origins=*",
            "--no-first-run",
            "--no-default-browser-check",
            "--new-window",
            "about:blank",
        ]
        try:
            # Plain Popen — NOT hidden_subprocess_kwargs(). Those kwargs set
            # SW_HIDE to suppress a CLI tool's console window, but Chrome is
            # a GUI app: SW_HIDE would launch its window hidden, so the page
            # loads invisibly. Chrome has no console to flash, so a normal
            # launch is correct and the window shows as expected.
            self._proc = subprocess.Popen(args)
            self._logger.event("web_chrome_launched", port=self._port)
            return True
        except Exception as exc:
            self._logger.exception("web_chrome_launch_failed", exc)
            return False

    def _debugger_up(self) -> bool:
        try:
            self._cdp_http("/json/version", timeout=1.0)
            return True
        except Exception:
            return False

    def _connect_page(self) -> bool:
        """Find a 'page' target and open a CDP websocket to it."""
        import websocket  # local import
        try:
            targets = self._cdp_http("/json")
        except Exception as exc:
            self._logger.exception("web_targets_failed", exc)
            return False
        page = None
        for t in targets:
            if t.get("type") == "page" and not str(t.get("url", "")).startswith("devtools://"):
                page = t
                break
        if page is None or not page.get("webSocketDebuggerUrl"):
            self._logger.event("web_no_page_target")
            return False
        ws_url = str(page["webSocketDebuggerUrl"]).replace("127.0.0.1", "localhost")
        try:
            self._ws = websocket.create_connection(ws_url, timeout=10)
            return True
        except Exception as exc:
            self._logger.exception("web_ws_connect_failed", exc)
            self._ws = None
            return False

    def ensure_chrome(self) -> bool:
        with self._lock:
            if self._ws is not None:
                return True
            if not self._debugger_up():
                if not self._launch_chrome():
                    return False
                # Wait for the debugger endpoint to come up.
                deadline = time.time() + 12.0
                while time.time() < deadline and not self._debugger_up():
                    time.sleep(0.3)
                if not self._debugger_up():
                    self._logger.event("web_debugger_timeout")
                    return False
            return self._connect_page()

    # ---- CDP command plumbing ----

    def _cdp(self, method: str, params: Optional[Dict[str, Any]] = None, timeout: float = 12.0) -> Dict[str, Any]:
        with self._lock:
            if self._ws is None:
                return {"error": {"message": "not connected"}}
            self._msg_id += 1
            mid = self._msg_id
            try:
                self._ws.send(json.dumps({"id": mid, "method": method, "params": params or {}}))
            except Exception as exc:
                self._logger.exception("web_cdp_send_failed", exc, method=method)
                self._ws = None
                return {"error": {"message": f"send failed: {exc}"}}
            deadline = time.time() + timeout
            while time.time() < deadline:
                try:
                    self._ws.settimeout(max(0.2, deadline - time.time()))
                    raw = self._ws.recv()
                except Exception:
                    break
                try:
                    msg = json.loads(raw)
                except Exception:
                    continue
                if msg.get("id") == mid:  # our response (ignore CDP events)
                    return msg
            return {"error": {"message": "cdp timeout"}}

    def _eval(self, expression: str, timeout: float = 12.0):
        """Runtime.evaluate returning a by-value JS result, or None on error."""
        resp = self._cdp(
            "Runtime.evaluate",
            {"expression": expression, "returnByValue": True, "awaitPromise": True},
            timeout=timeout,
        )
        if "error" in resp:
            return None
        return resp.get("result", {}).get("result", {}).get("value")

    # ---- public web operations ----

    def navigate(self, target: str) -> Dict[str, Any]:
        if not self.ensure_chrome():
            return {"status": "error", "error": "Chrome/CDP unavailable", "code": "cdp_unavailable"}
        url = _resolve_url(target)
        self._cdp("Page.enable")
        nav = self._cdp("Page.navigate", {"url": url})
        if "error" in nav:
            return {"status": "error", "error": str(nav["error"]), "code": "navigate_failed"}
        # Wait for the document to finish loading (poll readyState).
        deadline = time.time() + 15.0
        while time.time() < deadline:
            time.sleep(0.4)
            if self._eval("document.readyState") == "complete":
                break
        # Bring the page's window to the foreground so the user sees it.
        self._cdp("Page.bringToFront")
        info = self._eval("JSON.stringify({title:document.title,url:location.href})") or "{}"
        try:
            data = json.loads(info)
        except Exception:
            data = {}
        return {
            "status": "ok",
            "url": data.get("url", url),
            "title": data.get("title", ""),
        }

    def get_links(self, contains: Optional[str] = None, limit: int = 30) -> Dict[str, Any]:
        if not self.ensure_chrome():
            return {"status": "error", "error": "Chrome/CDP unavailable", "code": "cdp_unavailable"}
        js = (
            "JSON.stringify({host:location.host,links:Array.from(document.querySelectorAll('a'))"
            ".map(a=>({t:(a.innerText||a.textContent||'').trim().replace(/\\s+/g,' '),h:a.href}))"
            ".filter(l=>l.h && l.h.indexOf('http')===0 && l.t.length>1)})"
        )
        raw = self._eval(js)
        if not raw:
            return {"status": "ok", "links": [], "note": "no links found (page may still be loading)"}
        try:
            data = json.loads(raw)
            host = str(data.get("host", "")).lower()
            items = data.get("links", [])
        except Exception:
            host, items = "", []
        # On a search-engine results page, drop the engine's OWN nav / Maps /
        # account / utility links so "the Nth link" maps to a real result
        # website (for "restaurants near me" the top links were google Maps
        # links, so the 2nd link just bounced back to a google page).
        is_search = any(s in host for s in ("google.", "bing.", "duckduckgo.", "search.brave."))
        seen = set()
        all_links: List[Dict[str, Any]] = []
        for it in items:
            href = _unwrap_redirect(str(it.get("h", "")))
            text = str(it.get("t", ""))[:90]
            if not href or href in seen:
                continue
            if is_search and _is_engine_chrome(href):
                continue
            seen.add(href)
            all_links.append({"text": text, "url": href})

        # `contains` is a SOFT filter: narrow the list when it matches
        # (word-based + punctuation-insensitive, so "valentines" matches
        # "Valentine's"), but if nothing matches, return the FULL list so
        # the model can still pick the right link itself — never report
        # "no links" when links exist.
        note = None
        chosen = all_links
        needle = _norm(contains or "")
        if needle:
            nwords = [w for w in needle.split() if len(w) > 2]
            filtered = [
                l for l in all_links
                if needle in _norm(l["text"] + " " + l["url"])
                or (nwords and all(w in _norm(l["text"] + " " + l["url"]) for w in nwords))
            ]
            if filtered:
                chosen = filtered
            else:
                note = f"no links matched '{contains}' — showing all links so you can choose"

        links = [
            {"index": i + 1, "text": l["text"], "url": l["url"]}
            for i, l in enumerate(chosen[: max(1, limit)])
        ]
        self._logger.event(
            "web_links_extracted",
            host=host,
            total=len(all_links),
            returned=len(links),
            filtered_by=contains or None,
            sample=[l["text"] for l in links[:8]],
        )
        out: Dict[str, Any] = {"status": "ok", "links": links}
        if note:
            out["note"] = note
        return out

    def get_text(self, max_chars: int = 4000) -> Dict[str, Any]:
        if not self.ensure_chrome():
            return {"status": "error", "error": "Chrome/CDP unavailable", "code": "cdp_unavailable"}
        text = self._eval("document.body ? document.body.innerText : ''") or ""
        text = str(text)
        truncated = len(text) > max_chars
        return {"status": "ok", "text": text[:max_chars], "truncated": truncated}

    def click_text(self, text: str) -> Dict[str, Any]:
        if not self.ensure_chrome():
            return {"status": "error", "error": "Chrome/CDP unavailable", "code": "cdp_unavailable"}
        needle = json.dumps((text or "").strip().lower())
        js = (
            "(function(t){var els=Array.from(document.querySelectorAll("
            "'a,button,[role=button],input[type=submit],input[type=button]'));"
            "var el=els.find(function(e){var s=(e.innerText||e.value||'').trim().toLowerCase();"
            "return s && s.indexOf(t)!==-1;});"
            "if(el){el.scrollIntoView({block:'center'});el.click();return true;}return false;})(" + needle + ")"
        )
        ok = self._eval(js)
        if ok:
            return {"status": "ok", "clicked": text}
        return {"status": "error", "error": f"no clickable element matching '{text}'", "code": "no_match"}

    def scroll(self, to: str = "bottom") -> Dict[str, Any]:
        if not self.ensure_chrome():
            return {"status": "error", "error": "Chrome/CDP unavailable", "code": "cdp_unavailable"}
        to = (to or "bottom").strip().lower()
        if to == "bottom":
            js = "window.scrollTo(0, document.body.scrollHeight); Math.round(window.scrollY)"
        elif to == "top":
            js = "window.scrollTo(0, 0); 0"
        elif to == "up":
            js = "window.scrollBy(0, -Math.round(window.innerHeight*0.9)); Math.round(window.scrollY)"
        else:  # "down" / default
            to = "down"
            js = "window.scrollBy(0, Math.round(window.innerHeight*0.9)); Math.round(window.scrollY)"
        y = self._eval(js)
        self._cdp("Page.bringToFront")
        return {"status": "ok", "scrolled_to": to, "scroll_y": y}

    def fill(self, target: str, text: str) -> Dict[str, Any]:
        """Type into a form field identified by its label / placeholder /
        name / nearby text. Fires input+change events so the page reacts."""
        if not self.ensure_chrome():
            return {"status": "error", "error": "Chrome/CDP unavailable", "code": "cdp_unavailable"}
        needle = json.dumps((target or "").strip().lower())
        val = json.dumps(str(text))
        js = (
            "(function(t,v){function lbl(e){var p=[e.getAttribute('aria-label'),"
            "e.getAttribute('placeholder'),e.getAttribute('name'),e.id,"
            "e.getAttribute('title')];if(e.labels&&e.labels.length)p.push("
            "e.labels[0].innerText);return p.filter(Boolean).join(' ').toLowerCase();}"
            "var els=Array.from(document.querySelectorAll("
            "'input,textarea,select,[contenteditable=true]'));"
            "var el=els.find(function(e){return lbl(e).indexOf(t)!==-1;});"
            "if(!el)return false;el.focus();"
            "if(el.isContentEditable){el.innerText=v;}else{el.value=v;}"
            "el.dispatchEvent(new Event('input',{bubbles:true}));"
            "el.dispatchEvent(new Event('change',{bubbles:true}));return true;})("
            + needle + "," + val + ")"
        )
        ok = self._eval(js)
        if ok:
            return {"status": "ok", "filled": target}
        return {"status": "error", "error": f"no input matching '{target}'", "code": "no_match"}

    def wait_for(self, query: str, timeout_sec: float = 15.0) -> Dict[str, Any]:
        """Poll the page until a CSS selector matches OR visible text contains
        `query`. Use after an action that loads content asynchronously."""
        if not self.ensure_chrome():
            return {"status": "error", "error": "Chrome/CDP unavailable", "code": "cdp_unavailable"}
        q = json.dumps(str(query))
        js = (
            "(function(q){try{if(document.querySelector(q))return true;}catch(e){}"
            "return (document.body?document.body.innerText:'').toLowerCase()"
            ".indexOf(q.toLowerCase())!==-1;})(" + q + ")"
        )
        deadline = time.time() + max(1.0, min(60.0, float(timeout_sec)))
        while time.time() < deadline:
            if self._eval(js) is True:
                return {"status": "ok", "found": query}
            time.sleep(0.5)
        return {"status": "error", "error": f"'{query}' not found within {int(timeout_sec)}s", "code": "timeout"}

    def evaluate(self, expression: str) -> Dict[str, Any]:
        """Run arbitrary JavaScript in the page and return its value."""
        if not self.ensure_chrome():
            return {"status": "error", "error": "Chrome/CDP unavailable", "code": "cdp_unavailable"}
        value = self._eval(str(expression))
        # Keep the payload small.
        s = value if isinstance(value, (int, float, bool)) or value is None else str(value)
        if isinstance(s, str) and len(s) > 4000:
            s = s[:4000]
        return {"status": "ok", "result": s}

    def close(self) -> None:
        with self._lock:
            if self._ws is not None:
                try:
                    self._ws.close()
                except Exception:
                    pass
                self._ws = None
            # Leave the Chrome process running so the user keeps their tabs;
            # it's a separate debug instance and harmless to leave open.

# Author: Konstantin Markov
