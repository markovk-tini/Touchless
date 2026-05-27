"""Windows UI Automation (UIA) controller for the Live API agent.

Lets Iris read and operate ANY Windows app's UI through its accessibility
tree — element names + exact bounding rects + invoke/value patterns —
instead of guessing pixel coordinates from a screenshot. This is the
reliable path for "small apps without an API": dialogs, native settings,
installers, etc. nearly all expose a UIA tree.

Driven via `comtypes` (the UIAutomationCore typelib). Self-contained and
guarded: if UIA can't init on a machine, every call returns a clean error
and the agent falls back to screenshot+click_screen.

Clicking prefers the InvokePattern (no coordinates, most reliable),
falling back to SelectionItem/Toggle, then a real mouse click on the
element's center.
"""
from __future__ import annotations

import ctypes
from ctypes import wintypes
import os
import re
import time
from typing import Any, Dict, List, Optional

# UIA ControlType ids -> friendly label. Only the interactive ones the
# agent would act on (keeps the element list short + relevant).
_INTERACTIVE_TYPES = {
    50000: "button",
    50002: "checkbox",
    50003: "combobox",
    50004: "edit",
    50005: "link",
    50007: "listitem",
    50011: "menuitem",
    50013: "radio",
    50019: "tab",
    50024: "treeitem",
    50031: "splitbutton",
}

# Control types the auto-approve watcher must NEVER click — they are app
# chrome (menus, menu bar, tabs, title bar), not prompt buttons. Without this
# the watcher matched VS Code's "Run" MENU and clicked it every cycle.
_WATCH_EXCLUDE_TYPES = (
    50010,  # menu bar
    50011,  # menu item
    50019,  # tab item
    50037,  # title bar
)

# Pattern ids.
_INVOKE_PATTERN = 10000
_VALUE_PATTERN = 10002
_TOGGLE_PATTERN = 10015
_SELECTIONITEM_PATTERN = 10010
# Property ids.
_NAME_PROPERTY = 30005
_CONTROLTYPE_PROPERTY = 30003


def _norm(s: str) -> str:
    return " ".join(str(s or "").strip().lower().split())


# "❯ 1. Yes", "1. Yes", "1 Yes" — capture the digit that selects the Yes option.
_YES_NUM_RE = re.compile(r"(\d)\s*[\.\)]?\s*yes\b", re.IGNORECASE)

# Phrases that reliably identify an ACTIVE Claude Code approval prompt. Used to
# (a) decide Enter accepts the highlighted Yes default, and (b) locate the
# terminal element running Claude so we can give it keyboard focus before
# pressing. Kept module-level so both uses stay in sync.
_CLAUDE_PROMPT_MARKERS = (
    "do you want to proceed", "allow this", "do you want to",
    "esc to cancel", "tell claude what to do", "bash command?",
)


def _yes_keystroke(text: str) -> Optional[str]:
    """If `text` shows a KEYBOARD-answerable approval prompt (the kind Claude
    Code renders in its webview, which UIA often can't click), return the key
    that selects YES — a digit for a numbered prompt, 'y' for a y/n prompt, or
    'enter' to accept a highlighted default. Returns None when there's no clear
    prompt, so we never press keys speculatively."""
    low = (text or "").lower()
    # Numbered Claude Code prompt: "❯ 1. Yes ... 2. ... 3. No". Require a "no"
    # option too so this only fires on an actual yes/no choice.
    m = _YES_NUM_RE.search(text or "")
    if m and " no" in low and "yes" in low:
        return m.group(1)
    if "(y/n)" in low or "[y/n]" in low or " y/n" in low:
        return "y"
    # The terminal may expose the prompt as SEPARATE UIA elements ("1", "Yes",
    # "2", "No"), so the digit-adjacent-to-yes regex above can miss — but the
    # marker phrases reliably identify an active Claude Code prompt whose
    # highlighted default (option 1) is always Yes/Allow, so Enter accepts it.
    # Require "yes" so we never press Enter on a non-prompt.
    if any(mark in low for mark in _CLAUDE_PROMPT_MARKERS) and "yes" in low:
        return "enter"  # Yes is the highlighted default — Enter accepts it
    return None


class UiaController:
    def __init__(self, logger) -> None:
        self._logger = logger
        self._client = None
        self._uia = None
        self._init_failed = False
        self._auto_thread = None
        self._auto_stop = None  # threading.Event while auto-approve runs

    # ---- init / window resolution ----

    def _ensure(self) -> bool:
        if self._client is not None:
            return True
        if self._init_failed:
            return False
        try:
            import comtypes.client
            comtypes.client.GetModule("UIAutomationCore.dll")
            from comtypes.gen import UIAutomationClient as UIA
            self._uia = UIA
            self._client = comtypes.client.CreateObject(
                UIA.CUIAutomation, interface=UIA.IUIAutomation
            )
            return True
        except Exception as exc:
            self._init_failed = True
            self._logger.exception("uia_init_failed", exc)
            return False

    def _foreground_hwnd(self) -> int:
        try:
            return int(ctypes.windll.user32.GetForegroundWindow())
        except Exception:
            return 0

    def _window_pid(self, hwnd: int) -> int:
        try:
            pid = ctypes.c_ulong(0)
            ctypes.windll.user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
            return int(pid.value)
        except Exception:
            return 0

    def _process_name(self, pid: int) -> str:
        if not pid:
            return ""
        try:
            PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
            h = ctypes.windll.kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
            if not h:
                return ""
            try:
                buf = ctypes.create_unicode_buffer(260)
                size = ctypes.c_ulong(260)
                if ctypes.windll.kernel32.QueryFullProcessImageNameW(h, 0, buf, ctypes.byref(size)):
                    return os.path.basename(buf.value).lower()
            finally:
                ctypes.windll.kernel32.CloseHandle(h)
        except Exception:
            pass
        return ""

    def _window_text(self, hwnd: int) -> str:
        try:
            user32 = ctypes.windll.user32
            length = user32.GetWindowTextLengthW(hwnd)
            if length <= 0:
                return ""
            buf = ctypes.create_unicode_buffer(length + 1)
            user32.GetWindowTextW(hwnd, buf, length + 1)
            return buf.value or ""
        except Exception:
            return ""

    def _candidate_windows(self) -> List[int]:
        """Visible, titled, top-level windows NOT belonging to our own process
        (so we never target Touchless's own chat/main windows), in Z-order."""
        user32 = ctypes.windll.user32
        own = os.getpid()
        out: List[int] = []
        EnumProc = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p)

        def _cb(hwnd, _lparam):
            try:
                if not user32.IsWindowVisible(hwnd):
                    return True
                if self._window_pid(int(hwnd)) == own:
                    return True
                if not self._window_text(int(hwnd)).strip():
                    return True
                out.append(int(hwnd))
            except Exception:
                pass
            return True

        try:
            user32.EnumWindows(EnumProc(_cb), 0)
        except Exception:
            pass
        return out

    def _hwnd_for_title(self, title: str) -> int:
        """Resolve a window by matching `title` against either its title text
        OR its process name (so 'pycharm' finds pycharm64.exe even though the
        title is the project/file name). Skips Touchless's own windows."""
        needle = _norm(title)
        if not needle:
            return self._best_default_hwnd()
        for hwnd in self._candidate_windows():
            title_text = _norm(self._window_text(hwnd))
            proc = self._process_name(self._window_pid(hwnd))
            if needle in title_text or (proc and needle in proc):
                return hwnd
        return self._best_default_hwnd()

    def _best_default_hwnd(self) -> int:
        """Foreground window if it isn't ours; otherwise the top-most other
        app window (the app behind the assistant chat)."""
        fg = self._foreground_hwnd()
        if fg and self._window_pid(fg) != os.getpid():
            return fg
        candidates = self._candidate_windows()
        return candidates[0] if candidates else fg

    def _root_element(self, window_title: Optional[str]):
        hwnd = self._hwnd_for_title(window_title) if window_title else self._best_default_hwnd()
        if not hwnd:
            return None
        try:
            return self._client.ElementFromHandle(ctypes.c_void_p(hwnd))
        except Exception as exc:
            self._logger.exception("uia_element_from_handle_failed", exc, hwnd=hwnd)
            return None

    def _interactive_condition(self):
        c = self._client
        conds = [c.CreatePropertyCondition(_CONTROLTYPE_PROPERTY, t) for t in _INTERACTIVE_TYPES]
        cond = conds[0]
        for extra in conds[1:]:
            cond = c.CreateOrCondition(cond, extra)
        return cond

    # ---- public ops ----

    def list_elements(self, window_title: Optional[str] = None, limit: int = 40) -> Dict[str, Any]:
        if not self._ensure():
            return {"status": "error", "error": "UI Automation unavailable", "code": "no_uia"}
        root = self._root_element(window_title)
        if root is None:
            return {"status": "error", "error": "could not attach to a window", "code": "no_window"}
        try:
            window_name = root.CurrentName
        except Exception:
            window_name = ""
        try:
            found = root.FindAll(self._uia.TreeScope_Descendants, self._interactive_condition())
            total = found.Length
        except Exception as exc:
            self._logger.exception("uia_findall_failed", exc)
            return {"status": "error", "error": f"enumerate failed: {exc}", "code": "enumerate_failed"}
        elements: List[Dict[str, Any]] = []
        for i in range(total):
            if len(elements) >= max(1, limit):
                break
            try:
                e = found.GetElement(i)
                name = (e.CurrentName or "").strip()
                if not name:
                    continue
                r = e.CurrentBoundingRectangle
                if r.right <= r.left or r.bottom <= r.top:
                    continue  # zero-size / offscreen
                elements.append({
                    "index": len(elements) + 1,
                    "name": name[:80],
                    "type": _INTERACTIVE_TYPES.get(int(e.CurrentControlType), "control"),
                    "enabled": bool(e.CurrentIsEnabled),
                })
            except Exception:
                continue
        return {"status": "ok", "window": window_name, "elements": elements}

    # Affirmative approval targets. For these, if no clickable element is
    # found, we may answer the prompt with the KEYBOARD (Claude Code's webview
    # Yes/No can't be UIA-clicked). Gated to affirmatives so click('No') or
    # click('Cancel') never presses Yes.
    _APPROVE_WORDS = {
        "yes", "allow", "allow always", "accept", "approve", "keep",
        "continue", "ok", "okay", "proceed", "confirm",
    }

    def _click_or_press(self, target: str, window_title: Optional[str]) -> bool:
        """Try to activate `target` in priority order: (1) a REAL interactive
        button (invoke/select/toggle) — never a blind click, so terminal text
        named 'Yes' isn't mistaken for a button; (2) for affirmatives, the
        KEYBOARD (presses the prompt's Yes key) — this answers Claude Code's
        CLI/TUI and webview Yes/No that UIA can't click, and is tried BEFORE a
        blind text-click so a dead click can't mask the keystroke; (3) a
        full-tree named match + blind click as a last resort for non-standard
        webview buttons. Returns True on success."""
        is_affirm = _norm(target) in self._APPROVE_WORDS
        # (1) Real interactive control.
        el = self._find_by_name(target, window_title, exclude_types=_WATCH_EXCLUDE_TYPES)
        if el is not None and (self._invoke(el) or self._select(el) or self._toggle(el)):
            return True
        # (2) Keyboard (affirmative prompts only, so click('No')/('Cancel')
        # never presses Yes).
        if is_affirm:
            key = _yes_keystroke(self._read_window_text(window_title))
            if key and self._focus_and_press(window_title, key):
                return True
        # (3) Last resort: full-tree match + blind click.
        el = self._find_by_name(
            target, window_title, exclude_types=_WATCH_EXCLUDE_TYPES, use_full_tree=True
        )
        if el is not None and (
            self._invoke(el) or self._select(el) or self._toggle(el) or self._click_center(el)
        ):
            return True
        return False

    def click(self, target: str, window_title: Optional[str] = None) -> Dict[str, Any]:
        if not self._ensure():
            return {"status": "error", "error": "UI Automation unavailable", "code": "no_uia"}
        if self._click_or_press(target, window_title):
            return {"status": "ok", "clicked": target}
        # Cold-start: a just-opened window's element tree may not be fully
        # populated on the first query. Retry once.
        time.sleep(0.35)
        if self._click_or_press(target, window_title):
            return {"status": "ok", "clicked": target}
        return {"status": "error", "error": f"could not find/activate '{target}'", "code": "no_match"}

    def wait_and_click(
        self,
        target: str,
        *,
        timeout_sec: float = 30.0,
        window_title: Optional[str] = None,
        poll: float = 0.5,
    ) -> Dict[str, Any]:
        """Poll locally until an element named `target` appears (e.g. a 'Yes'
        button in a dialog that pops up), then click it. No tokens spent
        while waiting. Caps the wait so it can't block forever."""
        if not self._ensure():
            return {"status": "error", "error": "UI Automation unavailable", "code": "no_uia"}
        deadline = time.monotonic() + max(1.0, min(120.0, float(timeout_sec)))
        while time.monotonic() < deadline:
            if self._click_or_press(target, window_title):
                return {"status": "ok", "clicked": target, "waited": True}
            time.sleep(max(0.15, poll))
        return {
            "status": "error",
            "error": f"'{target}' did not appear within {int(timeout_sec)}s",
            "code": "timeout",
        }

    # Virtual-key map for wait_and_press (CLI/TUI prompt answering).
    _VK = {
        "enter": 0x0D, "return": 0x0D, "tab": 0x09, "esc": 0x1B, "escape": 0x1B,
        "space": 0x20, "up": 0x26, "down": 0x28, "left": 0x25, "right": 0x27,
        "backspace": 0x08, "delete": 0x2E,
    }

    def _press_key(self, name: str) -> bool:
        name = str(name).strip().lower()
        if not name:
            return False
        user32 = ctypes.windll.user32
        vk = self._VK.get(name)
        if vk is None and len(name) == 1:
            r = user32.VkKeyScanW(ord(name))
            if r != -1:
                vk = r & 0xFF
        if vk is None:
            return False
        try:
            user32.keybd_event(vk, 0, 0, 0)
            user32.keybd_event(vk, 0, 2, 0)
            return True
        except Exception:
            return False

    def _force_foreground(self, hwnd) -> None:
        """Genuinely bring `hwnd` to the foreground despite Windows' foreground
        lock, via the AttachThreadInput trick — so both the quick flick to the
        target AND the restore back to the user's window actually take. A plain
        SetForegroundWindow from a background process is silently ignored."""
        try:
            user32 = ctypes.windll.user32
            kernel32 = ctypes.windll.kernel32
            fg = user32.GetForegroundWindow()
            if fg == hwnd:
                return
            fg_tid = user32.GetWindowThreadProcessId(fg, None) if fg else 0
            our_tid = kernel32.GetCurrentThreadId()
            attached = False
            if fg_tid and fg_tid != our_tid:
                attached = bool(user32.AttachThreadInput(our_tid, fg_tid, True))
            user32.ShowWindow(hwnd, 9)  # SW_RESTORE if minimized
            user32.BringWindowToTop(hwnd)
            user32.SetForegroundWindow(hwnd)
            if attached:
                user32.AttachThreadInput(our_tid, fg_tid, False)
        except Exception:
            pass

    def _focus_and_press(self, window_title: Optional[str], keys) -> bool:
        """Quick-flick focus to the target window, press `keys`, then SNAP focus
        back to the user's previous window so they can keep working with minimal
        disruption. Used to answer KEYBOARD-driven prompts (Claude Code's
        webview Yes/No) UIA can't click — keystrokes only reach the FOCUSED
        window. Both the flick and the restore use _force_foreground so the
        restore reliably takes. Returns True if at least one key was sent."""
        keylist = keys if isinstance(keys, list) else [keys]
        try:
            hwnd = self._hwnd_for_title(window_title) if window_title else self._best_default_hwnd()
            if not hwnd:
                return False
            # Capture where the user actually is RIGHT NOW so we can return them.
            prev = self._foreground_hwnd()
            self._force_foreground(hwnd)
            time.sleep(0.12)  # let focus settle just enough for the key to land
            # Give the actual Claude TERMINAL element keyboard focus (not just
            # the window) so the key lands in the prompt even if another VS Code
            # pane was last focused.
            self._setfocus_prompt(window_title, _CLAUDE_PROMPT_MARKERS)
            time.sleep(0.06)
            pressed_any = False
            for k in keylist:
                if self._press_key(k):
                    pressed_any = True
                    time.sleep(0.04)
            # SNAP back to the user's window (robustly) so their typing continues
            # where they left off. Skip if it was us or the same window.
            if prev and prev != hwnd and self._window_pid(prev) != os.getpid():
                self._force_foreground(prev)
            return pressed_any
        except Exception:
            return False

    def _setfocus_prompt(self, window_title: Optional[str], needles) -> bool:
        """Give KEYBOARD FOCUS to the element running the active prompt (the VS
        Code terminal hosting Claude), so a subsequent 1/Enter lands there even
        if the user last clicked the editor. Finds the descendant whose
        text/name contains one of `needles`, then SetFocus on it — walking up to
        a keyboard-focusable ancestor if the text node itself can't take focus.
        Best-effort: returns True if focus was set, False to fall back to plain
        window-foreground behavior (no regression)."""
        root = self._root_element(window_title)
        if root is None:
            return False
        nlist = [_norm(n) for n in needles if _norm(n)]
        if not nlist:
            return False
        try:
            found = root.FindAll(self._uia.TreeScope_Descendants, self._client.CreateTrueCondition())
        except Exception:
            return False
        try:
            walker = self._client.ControlViewWalker
        except Exception:
            walker = None
        for i in range(min(found.Length, 300)):
            try:
                e = found.GetElement(i)
                hay = _norm((e.CurrentName or "") + " " + self._text_of(e))
                if not any(n in hay for n in nlist):
                    continue
                cur = e
                for _ in range(6):  # element, then up to 5 ancestors
                    if cur is None:
                        break
                    try:
                        cur.SetFocus()
                        return True
                    except Exception:
                        pass
                    if walker is None:
                        break
                    try:
                        cur = walker.GetParentElement(cur)
                    except Exception:
                        break
            except Exception:
                continue
        return False

    def _focus_terminal_and_press(self, window_title: Optional[str], keys, needles) -> bool:
        """Auto-approve variant of _focus_and_press: bring the window forward,
        give the CLAUDE TERMINAL keyboard focus, press `keys`, and KEEP focus
        there (no restore). Re-asserting focus on every watch cycle means
        repeated approvals always land in the terminal — the user opted into
        keeping it focused for hands-off auto-approve. Returns True if a key was
        sent."""
        keylist = keys if isinstance(keys, list) else [keys]
        try:
            hwnd = self._hwnd_for_title(window_title) if window_title else self._best_default_hwnd()
            if not hwnd:
                return False
            self._force_foreground(hwnd)
            time.sleep(0.10)
            self._setfocus_prompt(window_title, needles)
            time.sleep(0.06)
            pressed_any = False
            for k in keylist:
                if self._press_key(k):
                    pressed_any = True
                    time.sleep(0.04)
            return pressed_any
        except Exception:
            return False

    def _text_of(self, el) -> str:
        """Read an element's text via the UIA TextPattern (terminals, docs)."""
        try:
            pat = el.GetCurrentPattern(10014)  # UIA_TextPatternId
            if not pat:
                return ""
            tp = pat.QueryInterface(self._uia.IUIAutomationTextPattern)
            return tp.DocumentRange.GetText(20000) or ""
        except Exception:
            return ""

    def _read_window_text(self, window_title: Optional[str] = None) -> str:
        """Best-effort read of a window's visible text (e.g. a terminal's
        buffer) so we can watch for a prompt. Tries TextPattern on the root,
        then on likely text containers; falls back to interactive element
        names + the title so detection still works when TextPattern is absent."""
        root = self._root_element(window_title)
        if root is None:
            return ""
        best = self._text_of(root)
        try:
            found = root.FindAll(self._uia.TreeScope_Descendants, self._client.CreateTrueCondition())
            for i in range(min(found.Length, 200)):
                try:
                    e = found.GetElement(i)
                    ct = int(e.CurrentControlType)
                    if ct in (50004, 50020, 50030, 50033, 50025, 50026):  # edit/text/document/pane/custom/group
                        t = self._text_of(e)
                        if len(t) > len(best):
                            best = t
                    nm = e.CurrentName
                    if nm and nm not in best:
                        best += " " + nm
                except Exception:
                    continue
        except Exception:
            pass
        try:
            return (root.CurrentName or "") + " " + best
        except Exception:
            return best

    def wait_and_press(
        self,
        text: str,
        keys,
        *,
        timeout_sec: float = 30.0,
        window_title: Optional[str] = None,
        poll: float = 0.5,
    ) -> Dict[str, Any]:
        """Watch (locally, no tokens) until `text` appears in the window's
        content, then press `keys`. For terminal/CLI prompts (Claude Code's
        '1. Yes / 2. No', installers, REPLs) where there's no clickable
        button — you answer with a keystroke."""
        if not self._ensure():
            return {"status": "error", "error": "UI Automation unavailable", "code": "no_uia"}
        keylist = keys if isinstance(keys, list) else [keys]
        needle = _norm(text)
        if not needle:
            return {"status": "error", "error": "empty text", "code": "invalid_arguments"}
        deadline = time.monotonic() + max(1.0, min(180.0, float(timeout_sec)))
        while time.monotonic() < deadline:
            if needle in _norm(self._read_window_text(window_title)):
                time.sleep(0.2)
                # MUST focus the target window first — otherwise the keystroke
                # lands in whatever the user has focused (e.g. the Touchless
                # command box), not the prompt.
                ok = self._focus_and_press(window_title, keylist)
                return {
                    "status": "ok" if ok else "error",
                    "matched": text,
                    "pressed": keylist if ok else [],
                    "waited": True,
                }
            time.sleep(max(0.2, poll))
        return {
            "status": "error",
            "error": f"'{text}' did not appear within {int(timeout_sec)}s",
            "code": "timeout",
        }

    def start_auto_approve(
        self,
        targets,
        *,
        duration_sec: float = 600.0,
        window_title: Optional[str] = None,
        check_interval_sec: float = 5.0,
        idle_sec: float = 20.0,
        on_conclude=None,
        ocr_fallback=None,
    ) -> Dict[str, Any]:
        """Background watcher that keeps clicking any of `targets`
        ('Yes','Allow','Keep','Accept',...) as they appear, AND detects when
        the app has paused: no approve button to click AND the window text has
        stopped changing for `idle_sec` (so we don't conclude mid-edit while
        it's still streaming). On pause (or duration/stop) it calls
        on_conclude(reason, clicks, text_tail) — so Iris can read the final
        screen and tell the user whether it finished or is showing a prompt
        that isn't a simple approve. Non-blocking; own per-thread UIA client."""
        import threading
        if not self._ensure():
            return {"status": "error", "error": "UI Automation unavailable", "code": "no_uia"}
        self.stop_auto_approve()
        tlist = [str(t) for t in (targets if isinstance(targets, list) else [targets]) if str(t).strip()]
        if not tlist:
            return {"status": "error", "error": "no targets", "code": "invalid_arguments"}
        dur = max(5.0, min(1800.0, float(duration_sec)))
        check = max(1.0, min(30.0, float(check_interval_sec)))
        # "Paused" must persist across several checks, so keep idle >= ~3x the
        # check interval (else one slow poll could conclude prematurely).
        idle = max(4.0, min(300.0, float(idle_sec)), check * 3.0)
        stop = threading.Event()
        self._auto_stop = stop
        logger = self._logger

        def _loop():
            worker = UiaController(logger)  # own COM apartment / client on this thread
            if not worker._ensure():
                logger.event("auto_approve_no_uia")
                if on_conclude:
                    try:
                        on_conclude("no_uia", 0, "")
                    except Exception:
                        pass
                return
            start = time.monotonic()
            deadline = start + dur
            last_click = start
            last_change = start
            last_text = ""
            last_sig = ""
            clicks = 0
            reason = "duration"
            while not stop.is_set() and time.monotonic() < deadline:
                clicked = False
                # (1) A REAL interactive button (native "Yes"/"Allow"/"Keep"
                # dialogs). Interactive-only search + genuine activation patterns
                # (invoke/select/toggle) — deliberately NOT a blind coordinate
                # click, so plain TERMINAL TEXT named "Yes" (Claude Code's CLI
                # prompt) can't be mistaken for a button. Excludes menu/tab/
                # titlebar so we never hit VS Code's "Run" menu.
                if (time.monotonic() - last_click) > 1.2:
                    for t in tlist:
                        if stop.is_set():
                            break
                        el = worker._find_by_name(t, window_title, exclude_types=_WATCH_EXCLUDE_TYPES)
                        if el is not None and (
                            worker._invoke(el) or worker._select(el) or worker._toggle(el)
                        ):
                            last_click = time.monotonic()
                            clicks += 1
                            clicked = True
                            logger.event("auto_approve_clicked", target=t, total=clicks)
                            break
                # Read the window's text once for BOTH the keyboard prompt check
                # and idle detection. Strip DIGITS from the stability signature:
                # Claude's tab has ticking numbers (token counts, '2s ago') that
                # would otherwise reset the idle timer forever. Real new output
                # (words) still changes the signature.
                tail = worker._read_window_text(window_title)[-2000:]
                sig = re.sub(r"\d+", "#", _norm(tail))
                now = time.monotonic()
                if sig != last_sig:
                    last_sig = sig
                    last_text = tail
                    last_change = now
                # (2) KEYBOARD — Claude Code's CLI/TUI (and webview) Yes/No that
                # UIA cannot click. Tried BEFORE any full-tree text-click so a
                # dead click on terminal text can't mask the keystroke that
                # actually answers the prompt. Detection uses only the TAIL so a
                # prompt that already scrolled up doesn't keep re-firing; the
                # prompt collapses once answered, so _yes_keystroke then returns
                # None and we stop — yet an UNanswered prompt (failed press) is
                # still detected and retried next cycle.
                if not clicked and (now - last_click) > 1.2:
                    key = _yes_keystroke(tail[-700:])
                    if key and worker._focus_terminal_and_press(
                        window_title, key, _CLAUDE_PROMPT_MARKERS
                    ):
                        last_click = time.monotonic()
                        clicks += 1
                        clicked = True
                        logger.event("auto_approve_keypress", key=key, total=clicks)
                # (3) LAST RESORT — a named match ANYWHERE in the tree + blind
                # click, for non-standard webview buttons that expose no
                # interactive pattern AND no keyboard-answerable prompt text.
                if not clicked and (now - last_click) > 1.2:
                    for t in tlist:
                        if stop.is_set():
                            break
                        el = worker._find_by_name(
                            t, window_title, exclude_types=_WATCH_EXCLUDE_TYPES, use_full_tree=True
                        )
                        if el is not None and (
                            worker._invoke(el) or worker._select(el) or worker._click_center(el)
                        ):
                            last_click = time.monotonic()
                            clicks += 1
                            clicked = True
                            logger.event("auto_approve_clicked", target=t, total=clicks, via="fulltree")
                            break
                # (4) OCR FALLBACK — when UIA saw nothing. Claude Code's prompt
                # is often a webview/terminal canvas UIA can't read at all (no
                # element, no text). OCR reads pixels: it detects the prompt,
                # clicks its panel to focus, and presses the Yes key. This is
                # what makes the Claude Code EXTENSION's prompt answerable. Only
                # invoked when UIA found nothing, to keep OCR cost off the hot
                # path; resets the change timer so we don't conclude idle while
                # a prompt is actually present.
                if not clicked and ocr_fallback is not None and (now - last_click) > 1.2:
                    try:
                        if ocr_fallback():
                            last_click = time.monotonic()
                            last_change = last_click
                            clicks += 1
                            clicked = True
                            logger.event("auto_approve_ocr", total=clicks)
                    except Exception as exc:
                        logger.exception("auto_approve_ocr_failed", exc)
                if not clicked and (now - last_click) > idle and (now - last_change) > idle:
                    reason = "idle"
                    break
                stop.wait(check)
            if stop.is_set():
                reason = "stopped"
            logger.event("auto_approve_finished", clicks=clicks, reason=reason)
            if on_conclude:
                try:
                    on_conclude(reason, clicks, (last_text or "")[-1500:])
                except Exception as exc:
                    logger.exception("auto_approve_conclude_failed", exc)

        th = threading.Thread(target=_loop, name="IrisAutoApprove", daemon=True)
        th.start()
        self._auto_thread = th
        return {
            "status": "ok",
            "auto_approving": tlist,
            "duration_sec": int(dur),
            "idle_sec": int(idle),
            "note": "Watching in the background; I'll tell you when it pauses (done or needs you).",
        }

    def stop_auto_approve(self) -> Dict[str, Any]:
        if self._auto_stop is not None:
            self._auto_stop.set()
        self._auto_stop = None
        self._auto_thread = None
        return {"status": "ok", "stopped": True}

    def set_value(self, target: str, text: str, window_title: Optional[str] = None) -> Dict[str, Any]:
        if not self._ensure():
            return {"status": "error", "error": "UI Automation unavailable", "code": "no_uia"}
        el = self._find_by_name(target, window_title, prefer_types=(50004, 50003))
        if el is None:
            return {"status": "error", "error": f"no field named '{target}'", "code": "no_match"}
        try:
            pat = el.GetCurrentPattern(_VALUE_PATTERN)
            if pat:
                vp = pat.QueryInterface(self._uia.IUIAutomationValuePattern)
                vp.SetValue(str(text))
                return {"status": "ok", "set": target}
        except Exception as exc:
            self._logger.exception("uia_setvalue_failed", exc)
        # Fallback: focus + type.
        try:
            el.SetFocus()
            self._type_text(str(text))
            return {"status": "ok", "set": target, "method": "typed"}
        except Exception as exc:
            return {"status": "error", "error": f"could not set value: {exc}", "code": "set_failed"}

    # ---- element finding ----

    def _find_by_name(self, target: str, window_title: Optional[str], prefer_types=(), exclude_types=(), use_full_tree=False):
        root = self._root_element(window_title)
        if root is None:
            return None
        needle = _norm(target)
        try:
            # Default: only interactive controls (fast, precise). Full tree:
            # match ANY named element — needed for webview prompts (e.g. Claude
            # Code's Yes/No options) that aren't standard interactive controls
            # and so are invisible to the interactive-only search.
            cond = self._client.CreateTrueCondition() if use_full_tree else self._interactive_condition()
            found = root.FindAll(self._uia.TreeScope_Descendants, cond)
            total = found.Length
        except Exception:
            return None
        exact = None
        contains = None
        contains_pref = None
        exact_pref = None
        for i in range(total):
            try:
                e = found.GetElement(i)
                name = _norm(e.CurrentName)
                if not name:
                    continue
                ct = int(e.CurrentControlType)
                # Skip element kinds that are never a prompt button (menus,
                # menu bar, tabs, title bar) so the auto-approve watcher can't
                # click VS Code's "Run" menu / titlebar by accident.
                if exclude_types and ct in exclude_types:
                    continue
                if name == needle:
                    if prefer_types and ct in prefer_types and exact_pref is None:
                        exact_pref = e
                    elif exact is None:
                        exact = e
                elif needle and needle in name:
                    if prefer_types and ct in prefer_types and contains_pref is None:
                        contains_pref = e
                    elif contains is None:
                        contains = e
            except Exception:
                continue
        return exact_pref or exact or contains_pref or contains

    # ---- activation helpers ----

    def _invoke(self, el) -> bool:
        try:
            pat = el.GetCurrentPattern(_INVOKE_PATTERN)
            if not pat:
                return False
            pat.QueryInterface(self._uia.IUIAutomationInvokePattern).Invoke()
            return True
        except Exception:
            return False

    def _select(self, el) -> bool:
        try:
            pat = el.GetCurrentPattern(_SELECTIONITEM_PATTERN)
            if not pat:
                return False
            pat.QueryInterface(self._uia.IUIAutomationSelectionItemPattern).Select()
            return True
        except Exception:
            return False

    def _toggle(self, el) -> bool:
        try:
            pat = el.GetCurrentPattern(_TOGGLE_PATTERN)
            if not pat:
                return False
            pat.QueryInterface(self._uia.IUIAutomationTogglePattern).Toggle()
            return True
        except Exception:
            return False

    def _click_center(self, el) -> bool:
        # LAST-RESORT physical click. Guarded hard: a non-visible element
        # (collapsed/offscreen/just-named match) reports a zero
        # BoundingRectangle, and clicking its "center" would fling the cursor
        # to the top-left corner (0,0) and click there. We refuse degenerate
        # rects and restore the user's cursor afterward so we never steal the
        # pointer — important because the auto-approve watcher calls this on a
        # timer in the background.
        try:
            try:
                if bool(el.CurrentIsOffscreen):
                    return False
            except Exception:
                pass
            r = el.CurrentBoundingRectangle
            w = int(r.right) - int(r.left)
            h = int(r.bottom) - int(r.top)
            if w <= 1 or h <= 1:
                return False  # degenerate / hidden -> would jump to (0,0)
            x = int((r.left + r.right) / 2)
            y = int((r.top + r.bottom) / 2)
            if x <= 0 or y <= 0:
                return False  # off the top-left edge -> bogus target
            user32 = ctypes.windll.user32
            # Snapshot the real cursor so we can put it back after clicking.
            prev = wintypes.POINT()
            have_prev = bool(user32.GetCursorPos(ctypes.byref(prev)))
            user32.SetCursorPos(x, y)
            time.sleep(0.03)
            user32.mouse_event(0x0002, 0, 0, 0, 0)  # left down
            user32.mouse_event(0x0004, 0, 0, 0, 0)  # left up
            if have_prev:
                time.sleep(0.01)
                user32.SetCursorPos(int(prev.x), int(prev.y))  # restore pointer
            return True
        except Exception:
            return False

    def _type_text(self, text: str) -> None:
        # Simple unicode keystroke injection via SendInput-style mouse_event
        # isn't ideal; reuse the existing text controller path instead when
        # available. Here we fall back to per-char VkKeyScan keybd_event.
        user32 = ctypes.windll.user32
        for ch in text:
            vk = user32.VkKeyScanW(ord(ch))
            if vk == -1:
                continue
            code = vk & 0xFF
            shift = (vk >> 8) & 1
            if shift:
                user32.keybd_event(0x10, 0, 0, 0)
            user32.keybd_event(code, 0, 0, 0)
            user32.keybd_event(code, 0, 2, 0)
            if shift:
                user32.keybd_event(0x10, 0, 2, 0)

# Author: Konstantin Markov
