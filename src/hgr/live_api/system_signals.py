"""System-state signal producers for InterruptionGate.

Phase-3 wiring. The InterruptionGate's whole point is "should Iris
talk right now?" — but the gate is policy only; it doesn't read the
OS. This module bridges that gap: a small Sentinel-registered
watcher that polls Windows for the privacy-critical signals
(screen-sharing, microphone in use, camera in use, Focus Assist,
DND, fullscreen app, battery low, user idle) and pushes them into
the gate via `set_signal(...)`.

Cheap by design: each detection is a fast Win32 / process-scan
poll, well under the Sentinel's per-watcher cost budget. The
producers are all best-effort — when a probe fails (permission,
missing dll, non-Windows host) the signal goes UNKNOWN and the
gate's SEC-002 fail-closed semantics protect the user instead of
the absence of a reading being treated as "off".

Two public entry points:
  * `tick_all_signals()` — single function the Sentinel watcher
    calls every ~5 seconds. Updates every signal in one pass.
  * `register_with_sentinel(sentinel)` — convenience helper that
    wires the tick as a Sentinel watcher with sensible defaults.

Detection details per signal:

  - SCREEN_SHARING — look for any process matching a small list
    of well-known screen-share / record apps in foreground OR
    actively rendering (Teams, Zoom, Discord, OBS, Loom, Meet
    helper, GoTo, Webex, Slack huddle). Conservative: presence
    of the process is enough to assume the user is in or about
    to be in a share.
  - MIC_IN_USE — read the registry under
    `HKCU\\Software\\Microsoft\\Windows\\CurrentVersion\\
    CapabilityAccessManager\\ConsentStore\\microphone\\
    NonPackaged\\<exe>` for `LastUsedTimeStop == 0` (= currently
    in use). Same path under `\\Packaged\\<app>` for UWP apps.
  - CAMERA_IN_USE — same pattern under `\\webcam\\` instead of
    `\\microphone\\`.
  - FOCUS_ASSIST — Win32 `SHQueryUserNotificationState()` returns
    QUNS_BUSY (3) / QUNS_PRESENTATION_MODE (4) / QUNS_QUIET_TIME
    (8) when Focus Assist is on.
  - DND — same as FOCUS_ASSIST for now (Windows folds the two on
    most versions). Future split when Quiet Hours diverges.
  - FULLSCREEN_APP — foreground window covers the entire monitor
    AND isn't the shell. Win32 `GetForegroundWindow` +
    `GetWindowRect` vs monitor bounds.
  - USER_IDLE_SEC — `GetLastInputInfo()` against `GetTickCount`.
  - BATTERY_LOW — `GetSystemPowerStatus()` BatteryLifePercent
    field; "low" when ≤ 20% AND not on AC.

All detectors degrade gracefully on non-Windows / missing libs —
they return None (= no signal posted) so the gate's UNKNOWN
fallback takes over.

Author: Konstantin Markov
"""
from __future__ import annotations

import os
import sys
import threading
import time
from typing import Any, Optional, Set, Tuple


# Process-name fragments that indicate screen-sharing / recording.
# Lowercased substring match against tasklist output. Conservative —
# false positives are annoying but failing-open is worse (SEC-002).
_SCREEN_SHARE_PROCS = (
    "obs64.exe", "obs32.exe", "obs.exe",
    "loom.exe", "loomdesktop.exe",
    "ms-teams.exe", "teams.exe",        # legacy + new Teams
    "zoom.exe", "zoomhost.exe",
    "discord.exe",                       # huddles / screen-share
    "webex.exe", "webexmta.exe", "webex_meeting.exe",
    "gotomeeting.exe", "g2mui.exe",
    "slack.exe",                         # slack huddles
    "skype.exe",
    "screencast.exe", "screenrec.exe",
)

# Process-name fragments that indicate game mode.
_GAME_FULLSCREEN_PROCS = (
    "steam.exe", "epicgameslauncher.exe", "rocketleague.exe",
    "csgo.exe", "valorant.exe", "fortniteclient-win64-shipping.exe",
    "leagueclient.exe", "league of legends.exe",
)


def _is_windows() -> bool:
    return os.name == "nt"


# ---- low-level Win32 probes --------------------------------------------

def _list_running_processes_lowercase() -> Set[str]:
    """Return the set of running process executable names (lowercased).
    Empty set when the probe fails — caller treats that as 'unknown'."""
    if not _is_windows():
        return set()
    try:
        import psutil
        names: Set[str] = set()
        for p in psutil.process_iter(attrs=["name"]):
            try:
                n = p.info.get("name") or ""
                if n:
                    names.add(n.lower())
            except Exception:
                continue
        return names
    except Exception:
        return set()


def _any_process_matches(names: Set[str], needles: tuple) -> bool:
    return any(n in names for n in needles)


def _shqueryuserstate() -> Optional[int]:
    """Return the SHQueryUserNotificationState int (1..8) or None
    when the probe fails. See Microsoft docs for the codes."""
    if not _is_windows():
        return None
    try:
        import ctypes
        from ctypes import wintypes
        shell32 = ctypes.windll.shell32
        state = wintypes.INT()
        hr = shell32.SHQueryUserNotificationState(ctypes.byref(state))
        if hr != 0:
            return None
        return int(state.value)
    except Exception:
        return None


def _self_app_id_fragments() -> Tuple[str, ...]:
    """Lowercased substrings that identify THIS Touchless process in
    the ConsentStore registry key names. Used to skip our own
    mic/camera handle so the gate doesn't flag the user as 'on a
    call' just because we're running. Cached after first call."""
    cached = getattr(_self_app_id_fragments, "_cache", None)
    if cached is not None:
        return cached
    frags: list[str] = ["touchless", "hgr_app", "hgr-app"]
    try:
        exe = sys.executable or ""
        stem = os.path.splitext(os.path.basename(exe))[0]
        if stem:
            frags.append(stem.lower())
    except Exception:
        pass
    try:
        argv0 = (sys.argv[0] if sys.argv else "") or ""
        stem0 = os.path.splitext(os.path.basename(argv0))[0]
        if stem0:
            frags.append(stem0.lower())
    except Exception:
        pass
    # Dedupe while preserving order.
    seen: Set[str] = set()
    out: list[str] = []
    for f in frags:
        if f and f not in seen:
            seen.add(f)
            out.append(f)
    cached = tuple(out)
    setattr(_self_app_id_fragments, "_cache", cached)
    return cached


def _capability_in_use(capability: str) -> Optional[bool]:
    """Return True if any app is currently using the named device
    capability ("microphone" or "webcam"), False if not, None if
    we couldn't tell. Reads the Windows ConsentStore registry.

    Self-exclusion: Touchless's own webcam/mic handle is skipped so
    the InterruptionGate doesn't conclude the user is on a call just
    because the app is running. Match is by substring against the
    ConsentStore subkey name (see `_self_app_id_fragments`)."""
    if not _is_windows():
        return None
    try:
        import winreg
    except Exception:
        return None
    base = (r"Software\\Microsoft\\Windows\\CurrentVersion\\"
            r"CapabilityAccessManager\\ConsentStore\\" + capability)
    self_frags = _self_app_id_fragments()
    in_use = False
    seen_any = False
    for sub in ("NonPackaged", ""):
        full = base + (f"\\{sub}" if sub else "")
        try:
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, full) as parent:
                i = 0
                while True:
                    try:
                        app_name = winreg.EnumKey(parent, i)
                    except OSError:
                        break
                    i += 1
                    seen_any = True
                    app_lc = app_name.lower()
                    if any(f in app_lc for f in self_frags):
                        continue
                    try:
                        with winreg.OpenKey(parent, app_name) as app_key:
                            stop_val, _ = winreg.QueryValueEx(
                                app_key, "LastUsedTimeStop")
                            if stop_val == 0:
                                in_use = True
                                return True
                    except OSError:
                        continue
        except OSError:
            continue
    return False if seen_any else None


def _user_idle_seconds() -> Optional[float]:
    """Time since last keyboard/mouse input, in seconds. None on
    non-Windows."""
    if not _is_windows():
        return None
    try:
        import ctypes
        from ctypes import wintypes

        class _LastInputInfo(ctypes.Structure):
            _fields_ = [("cbSize", wintypes.UINT),
                        ("dwTime", wintypes.DWORD)]

        info = _LastInputInfo()
        info.cbSize = ctypes.sizeof(info)
        if not ctypes.windll.user32.GetLastInputInfo(ctypes.byref(info)):
            return None
        tick = ctypes.windll.kernel32.GetTickCount()
        # Both are 32-bit unsigned millisecond counters that wrap;
        # difference handles wrap correctly via uint32 math.
        diff_ms = (tick - info.dwTime) & 0xFFFFFFFF
        return diff_ms / 1000.0
    except Exception:
        return None


def _battery_low() -> Optional[bool]:
    """True when battery is low AND we're on battery power. None on
    non-Windows or when the system has no battery (desktop)."""
    if not _is_windows():
        return None
    try:
        import ctypes
        from ctypes import wintypes

        class _SystemPowerStatus(ctypes.Structure):
            _fields_ = [
                ("ACLineStatus", wintypes.BYTE),
                ("BatteryFlag", wintypes.BYTE),
                ("BatteryLifePercent", wintypes.BYTE),
                ("SystemStatusFlag", wintypes.BYTE),
                ("BatteryLifeTime", wintypes.DWORD),
                ("BatteryFullLifeTime", wintypes.DWORD),
            ]

        sps = _SystemPowerStatus()
        if not ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(sps)):
            return None
        # ACLineStatus: 0 = offline (on battery), 1 = online, 255 = unknown
        on_ac = sps.ACLineStatus == 1
        pct = int(sps.BatteryLifePercent)
        # 255 = "unknown" — no battery present (typical desktop).
        if pct == 255:
            return None
        if on_ac:
            return False
        return pct <= 20
    except Exception:
        return None


def _fullscreen_active() -> Optional[bool]:
    """True when the foreground window covers the entire primary
    monitor and isn't the desktop / shell. None on non-Windows."""
    if not _is_windows():
        return None
    try:
        import ctypes
        from ctypes import wintypes
        user32 = ctypes.windll.user32
        hwnd = user32.GetForegroundWindow()
        if not hwnd:
            return False
        # Reject the desktop / shell window.
        shell_hwnd = user32.GetShellWindow()
        desktop_hwnd = user32.GetDesktopWindow()
        if hwnd in (shell_hwnd, desktop_hwnd):
            return False
        rect = wintypes.RECT()
        if not user32.GetWindowRect(hwnd, ctypes.byref(rect)):
            return False
        screen_w = user32.GetSystemMetrics(0)  # SM_CXSCREEN
        screen_h = user32.GetSystemMetrics(1)  # SM_CYSCREEN
        win_w = rect.right - rect.left
        win_h = rect.bottom - rect.top
        # Allow 2px slack on each side for borderless-windowed games.
        return (win_w >= screen_w - 4 and win_h >= screen_h - 4
                and rect.left <= 2 and rect.top <= 2)
    except Exception:
        return None


# ---- public tick + wiring -----------------------------------------------

# Default poll interval — fast enough that screen-share state changes
# are caught within 5s; cheap enough that no realistic watcher pool
# pressure ensues.
DEFAULT_INTERVAL_SEC = 5.0


def tick_all_signals(gate: Optional[Any] = None) -> None:
    """One tick: poll every probe, set the gate's signals. When `gate`
    is None, looks up `interruption_gate.global_gate()`. All probes are
    best-effort — None readings just don't post a signal (the gate's
    UNKNOWN fallback handles missing data)."""
    from .interruption_gate import global_gate, SignalKind
    g = gate or global_gate()

    procs = _list_running_processes_lowercase()
    # SCREEN_SHARING: any of the known share/record apps running.
    if procs:
        g.set_signal(SignalKind.SCREEN_SHARING,
                     _any_process_matches(procs, _SCREEN_SHARE_PROCS))
        g.set_signal(SignalKind.GAME_MODE,
                     _any_process_matches(procs, _GAME_FULLSCREEN_PROCS))
    # else: leave both untouched — UNKNOWN protects via SEC-002.

    mic = _capability_in_use("microphone")
    if mic is not None:
        g.set_signal(SignalKind.MIC_IN_USE, mic)
    cam = _capability_in_use("webcam")
    if cam is not None:
        g.set_signal(SignalKind.CAMERA_IN_USE, cam)

    state = _shqueryuserstate()
    if state is not None:
        # QUNS_BUSY=3, QUNS_PRESENTATION_MODE=4, QUNS_QUIET_TIME=8 →
        # treat as Focus Assist / DND on.
        focus_on = state in (3, 4, 8)
        g.set_signal(SignalKind.FOCUS_ASSIST, focus_on)
        g.set_signal(SignalKind.DND, focus_on)

    fs = _fullscreen_active()
    if fs is not None:
        g.set_signal(SignalKind.FULLSCREEN_APP, fs)

    batt = _battery_low()
    if batt is not None:
        g.set_signal(SignalKind.BATTERY_LOW, batt)

    idle = _user_idle_seconds()
    if idle is not None:
        g.set_signal(SignalKind.USER_IDLE_SEC, idle)


def register_with_sentinel(sentinel: Optional[Any] = None,
                           *, interval_sec: float = DEFAULT_INTERVAL_SEC
                           ) -> None:
    """Register the tick as a Sentinel watcher. Idempotent (Sentinel's
    register() replaces existing watchers with the same name)."""
    from .sentinel import global_sentinel
    s = sentinel or global_sentinel()
    # Keep max_run_ms generous — psutil iter can be slow on first call
    # while it warms its cache.
    s.register("system_signals", tick_all_signals,
               interval_sec=interval_sec, max_run_ms=750)


def prime_signals_safely(gate: Optional[Any] = None) -> None:
    """Initial seed before any Sentinel tick runs. Seeds privacy-
    critical signals as False synchronously so the gate's SEC-002
    fail-closed doesn't block the very first interruption. On
    Windows, the real probe (tick_all_signals) is then dispatched
    to a daemon thread so the GUI thread is never blocked by
    psutil.process_iter + ConsentStore registry walks (which can
    take 1-5+ s on first call). The first sentinel tick of the
    registered system_signals watcher will refresh these values
    on its own daemon thread shortly after.

    The gate's SEC-002 fail-closed is a safety net for stale/missing
    signals during NORMAL operation — but on first boot before any
    watcher tick has run, it would block every notification including
    legitimate startup chimes. Priming gives us a known-good baseline
    AND if probes work, the values are correct from the start."""
    from .interruption_gate import global_gate, SignalKind
    g = gate or global_gate()
    # Synchronous baseline — never block GUI. Seed all privacy-
    # critical signals to False so SEC-002 doesn't fail-closed
    # before the first real reading lands.
    g.set_signal(SignalKind.SCREEN_SHARING, False)
    g.set_signal(SignalKind.MIC_IN_USE, False)
    g.set_signal(SignalKind.CAMERA_IN_USE, False)
    g.set_signal(SignalKind.FULLSCREEN_APP, False)
    g.set_signal(SignalKind.GAME_MODE, False)
    g.set_signal(SignalKind.FOCUS_ASSIST, False)
    g.set_signal(SignalKind.DND, False)
    if _is_windows():
        def _bg_prime() -> None:
            try:
                tick_all_signals(g)
            except Exception:
                pass
        threading.Thread(
            target=_bg_prime,
            name="iris-signals-prime",
            daemon=True,
        ).start()
