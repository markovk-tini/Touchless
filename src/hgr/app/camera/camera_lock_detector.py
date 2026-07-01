"""Detect and (optionally) release camera-holding processes.

The core problem this module solves:
Many Windows apps hold the camera's DirectShow filter graph open
even when they aren't actively displaying video. Razer Synapse 3,
RazerAxon, Razer Cortex, OBS Studio, Discord (when it thinks a
video call is imminent), Microsoft Teams, Zoom, Skype, Snap
Camera, Logi Capture / Logi G HUB, Windows Camera — any of these
running in the background can prevent Touchless from opening the
Kiyo Pro / BRIO / any webcam via ffmpeg-dshow or cv2.VideoCapture.

Touchless previously showed a generic "Camera connection lost.
Check the USB cable / make sure no other app is using the camera,
then press Stop and Start to retry" message. That's technically
correct but leaves the user hunting through Task Manager to find
what's holding the camera. This module surfaces the specific
processes AND optionally closes them, so the user gets a single
"click Yes to fix" experience.

Zero-setup principle: no new Python dependencies. Uses `tasklist`
(built into Windows since XP) via subprocess to enumerate running
processes. On non-Windows platforms every function no-ops safely.
"""
from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass


# Known camera-holding processes. Names matched against `tasklist`
# image name (case-insensitive). Grouped by publisher so the
# in-app dialog can render a "Razer suite (4 apps)" chip instead
# of listing them individually — cleaner UX than 4 lines of
# per-process entries.
#
# Each entry: (image_name, display_label, publisher_group).
# image_name may include the .exe suffix; matching is done
# case-insensitively and startswith to catch versioned filenames.
_KNOWN_CAMERA_HOLDERS: tuple[tuple[str, str, str], ...] = (
    # Razer suite — the biggest offender on setups with a Razer
    # peripheral (Kiyo webcams especially). Razer Synapse holds
    # the camera even when the Kiyo tile isn't visible.
    ("Razer Synapse 3.exe", "Razer Synapse 3", "razer"),
    ("Razer Central.exe", "Razer Central", "razer"),
    ("RazerCentralService.exe", "Razer Central Service", "razer"),
    ("Razer Synapse Service.exe", "Razer Synapse Service", "razer"),
    ("Razer Synapse Service Process.exe", "Razer Synapse Service Process", "razer"),
    ("RazerAxon.exe", "Razer Axon (streaming)", "razer"),
    ("RazerCortex.exe", "Razer Cortex", "razer"),
    # Screen recording / streaming — routinely capture cameras
    ("obs64.exe", "OBS Studio", "obs"),
    ("obs32.exe", "OBS Studio (32-bit)", "obs"),
    ("Streamlabs OBS.exe", "Streamlabs OBS", "obs"),
    ("StreamlabsDesktop.exe", "Streamlabs Desktop", "obs"),
    # Video call apps — hold the camera in "meeting-ready" state
    ("Discord.exe", "Discord", "chat"),
    ("ms-teams.exe", "Microsoft Teams", "chat"),
    ("Teams.exe", "Microsoft Teams (Classic)", "chat"),
    ("Zoom.exe", "Zoom", "chat"),
    ("Skype.exe", "Skype", "chat"),
    ("SnapCamera.exe", "Snap Camera", "chat"),
    # Windows built-in camera app
    ("WindowsCamera.exe", "Windows Camera", "windows"),
    # Logitech
    ("LogiCapture.exe", "Logi Capture", "logi"),
    ("lghub.exe", "Logi G HUB", "logi"),
    ("lghub_agent.exe", "Logi G HUB Agent", "logi"),
)

_PUBLISHER_LABELS: dict[str, str] = {
    "razer": "Razer suite",
    "obs": "OBS / Streamlabs",
    "chat": "Video-call apps",
    "windows": "Windows",
    "logi": "Logitech suite",
}


@dataclass(frozen=True)
class HoldingProcess:
    """One detected camera-holding process."""
    pid: int
    image_name: str
    display_label: str
    publisher: str


def _enumerate_running_processes() -> list[tuple[int, str]]:
    """Return list of (pid, image_name) for every running process.
    Uses Windows tasklist. On non-Windows returns empty list."""
    if not sys.platform.startswith("win"):
        return []
    try:
        # /fo csv gives a stable machine-readable format
        # /nh omits the header row
        raw = subprocess.check_output(
            ["tasklist", "/fo", "csv", "/nh"],
            stderr=subprocess.DEVNULL,
            timeout=4.0,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except (subprocess.SubprocessError, OSError):
        return []
    try:
        text = raw.decode("utf-8", errors="replace")
    except Exception:
        return []
    entries: list[tuple[int, str]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        # csv format: "image","pid","session","sessionid","memusage"
        parts = [p.strip('"') for p in line.split('","')]
        if len(parts) < 2:
            continue
        image = parts[0].strip('"').strip()
        try:
            pid = int(parts[1].strip('"').strip())
        except (ValueError, TypeError):
            continue
        if image:
            entries.append((pid, image))
    return entries


def detect_camera_holding_processes() -> list[HoldingProcess]:
    """Scan running processes for names in _KNOWN_CAMERA_HOLDERS.
    Returns matches ordered by publisher then display label so the
    UI can group them cleanly. Empty list on non-Windows or when
    nothing matches. Safe to call at any point — one shell out to
    tasklist (~150 ms typical)."""
    running = _enumerate_running_processes()
    if not running:
        return []
    lowered_running = [(pid, image.lower()) for pid, image in running]
    matches: list[HoldingProcess] = []
    seen_pids: set[int] = set()
    for image_name, display_label, publisher in _KNOWN_CAMERA_HOLDERS:
        needle = image_name.lower()
        for pid, running_image in lowered_running:
            if pid in seen_pids:
                continue
            if running_image == needle or running_image.startswith(needle.rstrip(".exe")):
                matches.append(HoldingProcess(
                    pid=pid,
                    image_name=image_name,
                    display_label=display_label,
                    publisher=publisher,
                ))
                seen_pids.add(pid)
    matches.sort(key=lambda p: (p.publisher, p.display_label))
    return matches


def summarize_processes_for_user(processes: list[HoldingProcess]) -> str:
    """Format a friendly summary string of detected processes,
    grouped by publisher. Used in log messages and dialog text."""
    if not processes:
        return "(none detected)"
    by_publisher: dict[str, list[str]] = {}
    for proc in processes:
        by_publisher.setdefault(proc.publisher, []).append(proc.display_label)
    parts: list[str] = []
    for pub, labels in by_publisher.items():
        group_label = _PUBLISHER_LABELS.get(pub, pub)
        # Deduplicate within a group (Discord runs multiple procs)
        unique_labels = sorted(set(labels))
        if len(unique_labels) == 1:
            parts.append(f"{group_label}: {unique_labels[0]}")
        else:
            parts.append(f"{group_label} ({len(unique_labels)} apps: {', '.join(unique_labels)})")
    return "; ".join(parts)


def close_processes(processes: list[HoldingProcess]) -> tuple[int, int, list[str]]:
    """Attempt to terminate the given processes. Uses taskkill /F /PID.
    Windows-only (no-op on other platforms).

    Returns (killed_count, failed_count, error_messages).

    NEVER call without user consent — this stops the user's other
    apps. The caller should have shown a dialog with an explicit
    Yes button before this fires.
    """
    if not sys.platform.startswith("win"):
        return (0, 0, ["Non-Windows platform — process kill is a no-op"])
    if not processes:
        return (0, 0, [])
    killed = 0
    failed = 0
    errors: list[str] = []
    for proc in processes:
        try:
            subprocess.check_call(
                ["taskkill", "/F", "/PID", str(proc.pid)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=4.0,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            killed += 1
        except subprocess.CalledProcessError:
            # Common cases: process already exited (race with our
            # enumeration), or lacking permissions (a Windows service
            # started by SYSTEM). Not fatal — record and move on.
            failed += 1
            errors.append(f"could not kill {proc.display_label} (PID {proc.pid})")
        except (subprocess.SubprocessError, OSError) as exc:
            failed += 1
            errors.append(f"{proc.display_label} (PID {proc.pid}): {exc!s}")
    return (killed, failed, errors)


# Author: Konstantin Markov
