"""Keep Windows' record of where Touchless is installed pointing at the app's
ACTUAL folder, so a user who moves the install (e.g. to another drive) still
updates correctly.

Why this exists: Touchless installs per-user via Inno Setup. Inno remembers the
install directory in the uninstall registry key under two values:
  * `Inno Setup: App Path`  — what UsePreviousAppDir reads, so the next run of
    the installer (in-app full-installer update OR the Microsoft Store's own
    "Update" button, both of which run the same Inno .exe) installs THERE.
  * `InstallLocation`       — the standard Windows value the Store version-check
    and Add/Remove Programs read.
Plus the uninstall/icon strings point at `unins000.exe` / `Touchless.exe` by
absolute path.

If the user moves the folder, all of those go stale (they still point at the
original location). The fast in-app "app-zip" update path is fine — it resolves
the RUNNING exe's directory and replaces files in place, like Chrome/VS Code do
— but the installer/Store path trusts the registry and would reinstall to the
OLD spot, leaving the moved copy un-updated (the exact "update failed" a user
hit after moving Touchless to another drive).

Fix: on every frozen launch, compare the running exe's directory to what the
registry says and, on a mismatch, rewrite the location values (and the
uninstall/icon strings when the referenced files are actually present next to
us). This makes a move transparent to every update path. HKCU only (per-user
install, no elevation); best-effort, never raises.

Author: Konstantin Markov
"""
from __future__ import annotations

import sys
from pathlib import Path

# Must match `AppId` in installers/windows/hgr_app.iss. Inno appends `_is1` to
# form the uninstall registry key name.
_INNO_APP_ID = "{2C4EE680-53F5-4D83-92A8-ADF4D2D8794E}_is1"
_UNINSTALL_SUBKEY = (
    r"Software\Microsoft\Windows\CurrentVersion\Uninstall" + "\\" + _INNO_APP_ID
)


def _running_install_dir() -> Path | None:
    """Directory of the running Touchless.exe (the real install location in a
    frozen build). None for source/dev runs, which have no install to heal."""
    if not getattr(sys, "frozen", False):
        return None
    try:
        return Path(sys.executable).resolve().parent
    except Exception:
        return None


def heal_install_location() -> bool:
    """Rewrite the uninstall-registry location values to the app's actual folder
    when they've gone stale (e.g. the user moved the install). Returns True if
    anything was rewritten, False otherwise. Windows + frozen only; HKCU only;
    never raises."""
    if not sys.platform.startswith("win"):
        return False
    install_dir = _running_install_dir()
    if install_dir is None:
        return False
    try:
        import winreg
    except Exception:
        return False

    current = str(install_dir)
    current_key = current.rstrip("\\").lower()

    try:
        key = winreg.OpenKey(
            winreg.HKEY_CURRENT_USER,
            _UNINSTALL_SUBKEY,
            0,
            winreg.KEY_READ | winreg.KEY_WRITE,
        )
    except FileNotFoundError:
        # No Inno per-user uninstall entry — not an installed build we manage
        # (dev copy, portable unzip, or a machine-wide/HKLM install we don't
        # touch from a per-user process). Nothing to heal.
        return False
    except Exception:
        return False

    try:
        # Prefer Inno's own App Path (the value UsePreviousAppDir consults);
        # fall back to the standard InstallLocation.
        stored = None
        for name in ("Inno Setup: App Path", "InstallLocation"):
            try:
                value, _ = winreg.QueryValueEx(key, name)
                if value:
                    stored = str(value)
                    break
            except Exception:
                continue

        if stored is not None and stored.rstrip("\\").lower() == current_key:
            return False  # already correct — the common case, read-only

        changed = False

        def _set(name: str, value: str) -> None:
            nonlocal changed
            try:
                winreg.SetValueEx(key, name, 0, winreg.REG_SZ, value)
                changed = True
            except Exception:
                pass

        # The two that steer future installs/updates.
        _set("Inno Setup: App Path", current)          # UsePreviousAppDir reads this
        _set("InstallLocation", current + "\\")         # Store + Add/Remove read this

        # Point the uninstall/icon strings at the files as they now sit — but
        # only if they're actually present here (the whole folder is normally
        # moved together, so they should be).
        exe = install_dir / "Touchless.exe"
        if exe.exists():
            _set("DisplayIcon", str(exe))
        unins = install_dir / "unins000.exe"
        if unins.exists():
            _set("UninstallString", f'"{unins}"')
            _set("QuietUninstallString", f'"{unins}" /SILENT')

        if changed:
            try:
                sys.stderr.write(
                    f"[install-location] healed stale registry path "
                    f"{stored!r} -> {current!r}\n"
                )
                sys.stderr.flush()
            except Exception:
                pass
        return changed
    finally:
        try:
            winreg.CloseKey(key)
        except Exception:
            pass
