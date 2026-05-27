"""Tiny persisted user preferences for the Live API assistant.

Stored as JSON under ~/Documents/Touchless/assistant_prefs.json (the same
per-user home LiveApiConfig logs to) so it survives auto-updates and lives
OUTSIDE the repo — never hardcode personal data in source. Override the
location with TOUCHLESS_PREFS_FILE.

Currently holds `default_email` (the address the email connector composes
from / the "your email" the model uses for "email myself").

Author: Konstantin Markov
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional


def _prefs_path() -> Path:
    override = os.environ.get("TOUCHLESS_PREFS_FILE")
    if override:
        return Path(override).expanduser()
    return Path.home() / "Documents" / "Touchless" / "assistant_prefs.json"


def load_prefs() -> Dict[str, Any]:
    path = _prefs_path()
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def set_pref(key: str, value: Any) -> bool:
    path = _prefs_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        prefs = load_prefs()
        prefs[key] = value
        path.write_text(json.dumps(prefs, indent=2), encoding="utf-8")
        return True
    except Exception:
        return False


def get_default_email() -> str:
    """The user's preferred email address. Pref file wins, then env, else "".
    """
    pref = (load_prefs().get("default_email") or "").strip()
    if pref:
        return pref
    return (os.environ.get("TOUCHLESS_DEFAULT_EMAIL") or "").strip()


def set_default_email(email: str) -> bool:
    return set_pref("default_email", (email or "").strip())
