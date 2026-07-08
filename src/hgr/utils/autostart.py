"""Windows auto-start-on-login helper.

Writes/removes a registry Run key under
HKCU\\Software\\Microsoft\\Windows\\CurrentVersion\\Run so Touchless
launches when the user signs in. We use HKCU (not HKLM) so the
toggle works without admin rights; the per-user install pattern
matches Inno's per-user output and Add/Remove Programs entry.

The value points at the installed Touchless.exe when shipped, or
the running python.exe + main module when running from source.
"""
from __future__ import annotations

import os
import platform
import sys
from pathlib import Path
from typing import Optional


_RUN_KEY_PATH = r"Software\Microsoft\Windows\CurrentVersion\Run"
_RUN_VALUE_NAME = "Touchless"

# macOS: a per-user LaunchAgent plist with RunAtLoad=True starts Touchless at
# login. Written to ~/Library/LaunchAgents/<label>.plist (per-user, no admin).
_MAC_LABEL = "com.touchless.app"


def is_supported() -> bool:
    return platform.system() in ("Windows", "Darwin")


def _mac_plist_path():
    return Path.home() / "Library" / "LaunchAgents" / f"{_MAC_LABEL}.plist"


def _mac_program_arguments() -> Optional[list]:
    """The argv the LaunchAgent runs at login. A frozen .app's sys.executable is
    Contents/MacOS/Touchless (run it directly); from source, launch the
    interpreter with -m hgr.app.main (dev convenience)."""
    exe = sys.executable
    if not exe:
        return None
    if getattr(sys, "frozen", False):
        return [exe]
    return [exe, "-m", "hgr.app.main"]


def _set_enabled_mac(enabled: bool) -> bool:
    try:
        import plistlib
        path = _mac_plist_path()
        if enabled:
            args = _mac_program_arguments()
            if not args:
                return False
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
            except Exception:
                return False
            plist = {
                "Label": _MAC_LABEL,
                "ProgramArguments": args,
                "RunAtLoad": True,
                "ProcessType": "Interactive",
            }
            with open(path, "wb") as fh:
                plistlib.dump(plist, fh)
            # Best-effort register now (also harmless if already loaded).
            try:
                import subprocess
                subprocess.run(["launchctl", "load", str(path)], capture_output=True, timeout=5)
            except Exception:
                pass
        else:
            try:
                import subprocess
                subprocess.run(["launchctl", "unload", str(_mac_plist_path())], capture_output=True, timeout=5)
            except Exception:
                pass
            try:
                _mac_plist_path().unlink(missing_ok=True)
            except Exception:
                pass
        return True
    except Exception:
        return False


def _resolve_launch_command() -> Optional[str]:
    """Build the command line Windows should run at login.

    When frozen (PyInstaller bundle), sys.executable is
    Touchless.exe directly -- a single quoted path is enough.
    When running from source, fall back to launching the
    interpreter with -m hgr; useful for dev, but the registry
    key is harmless either way on a machine that doesn't have
    the source tree.
    """
    if not is_supported():
        return None
    exe = sys.executable
    if not exe:
        return None
    # Frozen / installed path -- the launcher exe is what the user
    # actually wants Windows to run.
    if getattr(sys, "frozen", False):
        return f'"{exe}"'
    # Source-tree fallback. -m hgr.app.main matches the entry
    # point in pyproject.toml. Wrapped quotes handle paths with
    # spaces ("C:\HGR App v1.0.0\.venv\...").
    return f'"{exe}" -m hgr.app.main'


def set_enabled(enabled: bool) -> bool:
    """Add or remove the Run-key value. Returns True on success,
    False on any failure. Idempotent -- removing an absent value
    or adding an existing one is not an error."""
    if platform.system() == "Darwin":
        return _set_enabled_mac(enabled)
    if not is_supported():
        return False
    try:
        import winreg  # type: ignore
    except Exception:
        return False
    try:
        if enabled:
            command = _resolve_launch_command()
            if not command:
                return False
            with winreg.OpenKey(
                winreg.HKEY_CURRENT_USER,
                _RUN_KEY_PATH,
                0,
                winreg.KEY_SET_VALUE,
            ) as key:
                winreg.SetValueEx(key, _RUN_VALUE_NAME, 0, winreg.REG_SZ, command)
        else:
            try:
                with winreg.OpenKey(
                    winreg.HKEY_CURRENT_USER,
                    _RUN_KEY_PATH,
                    0,
                    winreg.KEY_SET_VALUE,
                ) as key:
                    winreg.DeleteValue(key, _RUN_VALUE_NAME)
            except FileNotFoundError:
                # Value already absent; success either way.
                pass
        return True
    except Exception:
        return False


def is_enabled() -> bool:
    """Read the registry to confirm the Run-key value is present.
    Used by the Settings checkbox to recover from external removal
    (e.g., msconfig disabling startup items)."""
    if platform.system() == "Darwin":
        try:
            return _mac_plist_path().exists()
        except Exception:
            return False
    if not is_supported():
        return False
    try:
        import winreg  # type: ignore
        with winreg.OpenKey(
            winreg.HKEY_CURRENT_USER,
            _RUN_KEY_PATH,
            0,
            winreg.KEY_READ,
        ) as key:
            value, _ = winreg.QueryValueEx(key, _RUN_VALUE_NAME)
            return bool(value)
    except FileNotFoundError:
        return False
    except Exception:
        return False
