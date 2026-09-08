"""Volume connector — the API-first path for system audio.

Wraps the existing VolumeController (pycaw / Core Audio). When this
connector is available, iris sets the system volume / mute with one
deterministic call instead of nudging the volume keys or clicking the
tray mixer — no screenshots, no guessing the current level.

VolumeController works with a 0..1 scalar internally; the model speaks
in 0-100 percent, so we convert at the boundary.

Author: Konstantin Markov
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from .base import Connector, connector_result


class VolumeConnector(Connector):
    id = "volume"

    def __init__(self, controller: Optional[Any] = None) -> None:
        self._controller = controller

    def _ctrl(self):
        if self._controller is None:
            from ...debug.volume_controller import VolumeController
            self._controller = VolumeController()
        return self._controller

    def available(self) -> bool:
        try:
            return bool(getattr(self._ctrl(), "available", False))
        except Exception:
            return False

    def tools(self) -> List[Dict[str, Any]]:
        def fn(name: str, desc: str, props: Dict[str, Any] | None = None,
               required: List[str] | None = None) -> Dict[str, Any]:
            return {
                "type": "function",
                "name": name,
                "description": desc,
                "parameters": {
                    "type": "object",
                    "properties": props or {},
                    "required": required or [],
                    "additionalProperties": False,
                },
            }

        return [
            fn("volume_get",
               "Get the current system volume (0-100 percent) and mute state. "
               "Prefer this over reading the volume from a screenshot."),
            fn("volume_set",
               "Set the system volume to a percent (0-100). Prefer this over "
               "pressing the volume keys repeatedly.",
               {"percent": {"type": "integer", "description": "Volume 0-100."}},
               ["percent"]),
            fn("volume_mute",
               "Mute or unmute the system audio.",
               {"muted": {"type": "boolean",
                          "description": "True to mute, False to unmute."}},
               ["muted"]),
            fn("volume_toggle_mute", "Toggle the system mute state."),
            # ---- per-app mixer (pycaw IAudioSessionManager2) -----------
            fn("volume_list_apps",
               "List every app currently producing or playing audio: "
               "name, exe, volume (0-100), muted. Use BEFORE volume_set_app "
               "to confirm the right session name (apps appear as their "
               "exe name — chrome.exe, Discord.exe, Spotify.exe, etc.)."),
            fn("volume_set_app",
               "Set the volume (0-100) for ONE app's audio session. "
               "Matches by substring against the exe name (case-insensitive), "
               "so 'discord' / 'chrome' / 'spotify' work. If multiple sessions "
               "match, all of them are adjusted (e.g. a browser with many "
               "tabs has one session per audio source). Returns the list of "
               "sessions actually changed.",
               {"app": {"type": "string",
                        "description": "App name or exe name substring "
                                       "(e.g. 'discord', 'chrome.exe')."},
                "percent": {"type": "integer",
                            "description": "0-100. 0 = silent."}},
               ["app", "percent"]),
            fn("volume_mute_app",
               "Mute or unmute one app's audio session by app-name "
               "substring (same matching as volume_set_app).",
               {"app": {"type": "string"},
                "muted": {"type": "boolean"}},
               ["app", "muted"]),
        ]

    def execute(self, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        c = self._ctrl()
        if name == "volume_get":
            level = c.get_level()
            mute = c.get_mute()
            return connector_result(
                "ok",
                percent=(round(level * 100) if isinstance(level, (int, float)) else None),
                muted=mute,
            )
        if name == "volume_set":
            # Tolerant arg parsing — the Tier-2 LLM planner often emits the
            # value under a different key ('level', 'value', 'volume') or
            # with a trailing '%' / decimal, which previously failed the
            # strict int() with "percent must be an integer 0-100" and the
            # volume_set step was the only one in a multi-step plan to err.
            raw = (args.get("percent") if args.get("percent") is not None
                   else args.get("level") if args.get("level") is not None
                   else args.get("value") if args.get("value") is not None
                   else args.get("volume"))
            if isinstance(raw, str):
                raw = raw.strip().rstrip("%").strip()
            try:
                pct = int(round(float(raw)))
            except (TypeError, ValueError):
                return connector_result(
                    "error",
                    error=("percent must be a number 0-100 (got "
                           f"{args.get('percent')!r})"))
            pct = max(0, min(100, pct))
            return connector_result("ok" if c.set_level(pct / 100.0) else "error", percent=pct)
        if name == "volume_mute":
            muted = bool(args.get("muted"))
            return connector_result("ok" if c.set_mute(muted) else "error", muted=muted)
        if name == "volume_toggle_mute":
            result = c.toggle_mute()
            return connector_result("ok" if result is not None else "error", muted=result)
        if name == "volume_list_apps":
            sessions, err = _list_audio_sessions()
            if err:
                return connector_result("error", error=err,
                                        code="pycaw_unavailable")
            return connector_result("ok",
                                    sessions=sessions,
                                    count=len(sessions))
        if name == "volume_set_app":
            app = str(args.get("app") or "").strip().lower()
            if not app:
                return connector_result("error", error="'app' is required")
            try:
                pct = int(round(float(args.get("percent"))))
            except (TypeError, ValueError):
                return connector_result(
                    "error",
                    error="percent must be a number 0-100 (got "
                          f"{args.get('percent')!r})")
            pct = max(0, min(100, pct))
            changed, err = _set_session_volume(app, pct / 100.0)
            if err:
                return connector_result("error", error=err,
                                        code="pycaw_unavailable")
            if not changed:
                return connector_result(
                    "error",
                    error=f"no audio session matches {app!r}. Call "
                          "volume_list_apps to see what's playing.",
                    code="no_match")
            return connector_result("ok", changed=changed, percent=pct)
        if name == "volume_mute_app":
            app = str(args.get("app") or "").strip().lower()
            if not app:
                return connector_result("error", error="'app' is required")
            muted = bool(args.get("muted"))
            changed, err = _set_session_mute(app, muted)
            if err:
                return connector_result("error", error=err,
                                        code="pycaw_unavailable")
            if not changed:
                return connector_result(
                    "error",
                    error=f"no audio session matches {app!r}.",
                    code="no_match")
            return connector_result("ok", changed=changed, muted=muted)
        return connector_result("error", error=f"unknown volume tool: {name}", code="no_handler")


# ---- per-app mixer helpers (pycaw) -----------------------------------------


def _list_audio_sessions():
    """Returns ([{name, exe, pid, volume_pct, muted}, ...], err)."""
    try:
        from pycaw.pycaw import AudioUtilities  # type: ignore
    except Exception:
        return [], ("pycaw not installed (pip install pycaw) — "
                    "per-app volume unavailable")
    out = []
    try:
        sessions = AudioUtilities.GetAllSessions()
    except Exception as exc:
        return [], f"pycaw enumeration failed: {type(exc).__name__}: {exc}"
    for s in sessions or []:
        try:
            volume = s.SimpleAudioVolume
            if volume is None:
                continue
            proc = s.Process
            exe = ""
            pid = 0
            if proc is not None:
                try:
                    exe = (proc.name() or "") if hasattr(proc, "name") else ""
                except Exception:
                    exe = ""
                try:
                    pid = int(proc.pid) if hasattr(proc, "pid") else 0
                except Exception:
                    pid = 0
            name = (s.DisplayName or "").strip() or exe or "system"
            try:
                v = float(volume.GetMasterVolume())
            except Exception:
                v = 0.0
            try:
                m = bool(volume.GetMute())
            except Exception:
                m = False
            out.append({
                "name": name,
                "exe": exe,
                "pid": pid,
                "volume_pct": int(round(v * 100)),
                "muted": m,
            })
        except Exception:
            # A single bad session must not break the listing.
            continue
    return out, None


def _set_session_volume(app_query: str, level_0_to_1: float):
    """Returns (list[changed_session_names], err)."""
    try:
        from pycaw.pycaw import AudioUtilities  # type: ignore
    except Exception:
        return [], "pycaw not installed"
    changed = []
    try:
        sessions = AudioUtilities.GetAllSessions()
    except Exception as exc:
        return [], f"pycaw enumeration failed: {exc}"
    for s in sessions or []:
        if _session_matches(s, app_query):
            try:
                if s.SimpleAudioVolume is not None:
                    s.SimpleAudioVolume.SetMasterVolume(
                        float(max(0.0, min(1.0, level_0_to_1))), None)
                    changed.append(_session_label(s))
            except Exception:
                continue
    return changed, None


def _set_session_mute(app_query: str, muted: bool):
    try:
        from pycaw.pycaw import AudioUtilities  # type: ignore
    except Exception:
        return [], "pycaw not installed"
    changed = []
    try:
        sessions = AudioUtilities.GetAllSessions()
    except Exception as exc:
        return [], f"pycaw enumeration failed: {exc}"
    for s in sessions or []:
        if _session_matches(s, app_query):
            try:
                if s.SimpleAudioVolume is not None:
                    s.SimpleAudioVolume.SetMute(1 if muted else 0, None)
                    changed.append(_session_label(s))
            except Exception:
                continue
    return changed, None


def _session_matches(session, query: str) -> bool:
    q = (query or "").strip().lower()
    if not q:
        return False
    name = ((session.DisplayName or "") or "").lower()
    exe = ""
    try:
        if session.Process is not None and hasattr(session.Process, "name"):
            exe = (session.Process.name() or "").lower()
    except Exception:
        exe = ""
    if not name and not exe:
        return False
    if q in name or q in exe:
        return True
    # Allow matching on the exe stem ('chrome' matches 'chrome.exe').
    stem = exe.rsplit(".", 1)[0] if "." in exe else exe
    return bool(stem) and (q == stem or q in stem)


def _session_label(session) -> str:
    try:
        if session.Process is not None and hasattr(session.Process, "name"):
            n = session.Process.name() or ""
            if n:
                return n
    except Exception:
        pass
    return (session.DisplayName or "session").strip()
