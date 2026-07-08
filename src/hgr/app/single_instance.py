"""Prevent two Touchless processes from running concurrently.

Two Touchless instances would fight over the camera, microphone,
and any QR-paired phone server (the second app can't bind to the
already-listening port). The user typically hits this by double-
clicking the desktop shortcut while one is already running, or
when the auto-updater respawns and the previous process exits a
second too late.

Strategy: a Win32 named mutex held for the lifetime of the
process. If the mutex already exists, a Touchless is already up;
the new instance bails out immediately and tries to focus the
existing window via FindWindow + SetForegroundWindow.

Why a Win32 mutex (not Qt's QSharedMemory): mutexes are reliably
torn down when the holding process exits — even if it crashed —
because the kernel cleans up on handle close. QSharedMemory leaks
its segment when the process is killed unexpectedly, leaving the
"already running" check stuck in the True state until reboot.
"""
from __future__ import annotations

import ctypes
import sys
from ctypes import wintypes

# A mutex name globally unique to Touchless. The "Local\\" prefix
# scopes it to the current Windows session — different users on
# the same machine each get their own Touchless instance, which
# matches our per-user install model.
_MUTEX_NAME = "Local\\Touchless_SingleInstance_2C4EE680"
_ERROR_ALREADY_EXISTS = 183
_SW_SHOWNORMAL = 1
_SW_RESTORE = 9

# Mapping between command-line arg (used by the Jump List shortcuts)
# and the registered window-message name we use to deliver the action
# to the running instance. RegisterWindowMessageW returns the same
# integer ID for a given name in every process, so the bailing
# second instance and the running first instance see the same ID.
_ACTION_MESSAGE_NAMES = {
    "--touchless-pause-30":  "Touchless_Action_Pause30_2C4EE680",
    "--touchless-settings":  "Touchless_Action_Settings_2C4EE680",
    "--touchless-quit":      "Touchless_Action_Quit_2C4EE680",
}


_handle: int | None = None
# macOS: an open file descriptor holding an exclusive fcntl.flock for the
# process lifetime. Held in a module global so it isn't garbage-collected (which
# would release the lock). The kernel releases the flock automatically when the
# process exits — even on crash — so there is no stale-lock problem.
_mac_lock_fd = None


def _acquire_mac() -> bool:
    """macOS single-instance via an exclusive fcntl.flock on a per-user lockfile.
    Returns True if we got the lock (only instance), False if another Touchless
    already holds it. Permissive (True) on any unexpected error — never block a
    legitimate launch."""
    global _mac_lock_fd
    try:
        import fcntl
        import os
        from pathlib import Path

        lock_dir = Path.home() / "Library" / "Application Support" / "Touchless"
        try:
            lock_dir.mkdir(parents=True, exist_ok=True)
            lock_path = lock_dir / "touchless.lock"
        except Exception:
            lock_path = Path("/tmp/touchless_singleinstance.lock")
        fd = open(lock_path, "a+")
        try:
            fcntl.flock(fd.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            # Lock held by another live Touchless -> this is a second instance.
            try:
                fd.close()
            except Exception:
                pass
            return False
        # Got it — hold the fd open for the whole process lifetime.
        _mac_lock_fd = fd
        try:
            fd.seek(0)
            fd.truncate()
            fd.write(str(os.getpid()))
            fd.flush()
        except Exception:
            pass
        return True
    except Exception:
        return True


def action_message_id(action_arg: str) -> int | None:
    """Resolve the Win32 RegisterWindowMessage ID for one of our
    Jump-List action args. Returns None if the arg isn't one we
    recognise or if the registration fails. Both the sender and the
    receiver call this; the result is per-process-stable but
    consistent across processes on the same OS instance."""
    if sys.platform != "win32":
        return None
    name = _ACTION_MESSAGE_NAMES.get(action_arg)
    if name is None:
        return None
    try:
        user32 = ctypes.windll.user32
        user32.RegisterWindowMessageW.argtypes = [wintypes.LPCWSTR]
        user32.RegisterWindowMessageW.restype = wintypes.UINT
        msg_id = user32.RegisterWindowMessageW(name)
        return int(msg_id) if msg_id else None
    except Exception:
        return None


def action_message_id_map() -> dict[int, str]:
    """Build {win32-message-id: action-arg} so the running instance
    can install a single nativeEvent filter that matches incoming
    messages back to the action it should perform."""
    out: dict[int, str] = {}
    for arg in _ACTION_MESSAGE_NAMES:
        msg_id = action_message_id(arg)
        if msg_id is not None:
            out[msg_id] = arg
    return out


def acquire(args: list[str] | None = None) -> bool:
    """Try to acquire the single-instance lock. Returns True if
    this is the only Touchless instance, False if another is
    already running. Caller should exit on False.

    When False is about to be returned and `args` contains one of
    the Jump-List action flags, post the corresponding Win32
    message to the running instance's main window before bailing
    so the user's right-click-task fires the same action as the
    tray menu."""
    global _handle
    if sys.platform == "darwin":
        return _acquire_mac()
    if sys.platform != "win32":
        return True
    try:
        kernel32 = ctypes.windll.kernel32
        kernel32.CreateMutexW.argtypes = (
            ctypes.c_void_p,
            wintypes.BOOL,
            wintypes.LPCWSTR,
        )
        kernel32.CreateMutexW.restype = wintypes.HANDLE
        kernel32.GetLastError.restype = wintypes.DWORD
        # Initial owner = False so we can probe the GetLastError
        # afterwards reliably.
        _handle = kernel32.CreateMutexW(None, False, _MUTEX_NAME)
        if not _handle:
            # Mutex creation outright failed — be permissive and
            # let the app continue rather than blocking launch.
            return True
        if kernel32.GetLastError() == _ERROR_ALREADY_EXISTS:
            # Mutex existed before our call: another Touchless is
            # already running. Forward any Jump-List action arg to
            # it (PostMessageW), focus its window for visibility,
            # then signal "don't start".
            if args:
                for arg in args:
                    if arg in _ACTION_MESSAGE_NAMES:
                        _post_action_to_running(arg)
                        break
            _focus_existing_window()
            return False
        return True
    except Exception:
        return True


def _post_action_to_running(action_arg: str) -> None:
    """Fire-and-forget PostMessageW to the running Touchless's main
    window with the action's registered message ID. The running
    instance's nativeEvent filter (installed on MainWindow) picks
    it up and dispatches the same handler the tray menu uses."""
    msg_id = action_message_id(action_arg)
    if msg_id is None:
        return
    try:
        user32 = ctypes.windll.user32
        user32.FindWindowW.argtypes = [wintypes.LPCWSTR, wintypes.LPCWSTR]
        user32.FindWindowW.restype = wintypes.HWND
        user32.PostMessageW.argtypes = [
            wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM,
        ]
        user32.PostMessageW.restype = wintypes.BOOL
        hwnd = user32.FindWindowW(None, "Touchless")
        if hwnd:
            user32.PostMessageW(hwnd, msg_id, 0, 0)
    except Exception:
        pass


def _focus_existing_window() -> None:
    """Best-effort raise of the running Touchless's main window."""
    try:
        user32 = ctypes.windll.user32
        # Touchless titles its main window "Touchless" (see
        # MainWindow.setWindowTitle). FindWindow with a NULL class
        # name searches by window title only.
        hwnd = user32.FindWindowW(None, "Touchless")
        if not hwnd:
            return
        # If the window is minimized, restore it. Then bring to top.
        user32.ShowWindow(hwnd, _SW_RESTORE)
        user32.SetForegroundWindow(hwnd)
    except Exception:
        pass

# Author: Konstantin Markov
