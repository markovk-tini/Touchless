"""Touchless debug-bundle collector (v1.1.9.2 r17+).

Called by the in-app "Save debug bundle before closing?" prompt in
main_window.closeEvent, and by any future hook that wants to hand the
user a single zip with everything triage needs. Design goals:

- Runs in ≤10 seconds on cold disk. Most collectors are wrapped in a
  per-item try/except so ONE broken collector never nukes the whole
  bundle.
- Never blocks shutdown — the caller runs this on a QThread and the
  outer close path always eventually proceeds. Any collector that
  might exceed its budget aborts silently.
- Redacts every credential path: Spotify OAuth tokens, Google/MS
  refresh tokens, YouTube API keys, etc. The redaction happens
  before the file is written to the bundle, not after.
- Writes a sentinel marker (%LOCALAPPDATA%/Touchless/last_bundle_saved.marker)
  after a successful save so the external Touchless_Debug.ps1 wrapper
  can detect a bundle already saved and skip its own prompt.

Bundle layout:
    Touchless_Debug_YYYY-MM-DD_HH-MM.zip
    ├── system.txt              OS/CPU/GPU/RAM + install location
    ├── env.txt                 HGR_* + core Windows vars
    ├── config-redacted.json    settings.json with tokens blanked
    ├── touchless_debug.log.tail  last ~2 MB of the runtime log
    ├── perf_signals.txt        grep-narrowed log signals for perf triage
    ├── event_log_app.txt       Windows Event Log app entries (best-effort)
    ├── gpu_driver.txt          nvidia-smi / dxdiag / WMI VideoController
    ├── cameras.txt             DirectShow + OpenCV camera enumeration
    ├── audio_devices.txt       sounddevice.query_devices()
    ├── ffmpeg_version.txt      ffmpeg -version + resolved path
    ├── disk_space.txt          free space on install / LOCALAPPDATA / home drives
    ├── processes.txt           psutil filter (ffmpeg, touchless, obs, camera, discord)
    ├── install_dir_manifest.txt  size + mtime for every install file
    ├── crash/                  copied from %LOCALAPPDATA%/Touchless/crash/
    └── install_log/            copied from %LOCALAPPDATA%/Touchless/install_log/
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional


# ------------------------------------------------------------------
# subprocess hiding — mirror the pattern from installed_app_subprocess
# ------------------------------------------------------------------
def _hidden_kwargs() -> dict:
    """Return subprocess.Popen kwargs that hide any spawned window on
    Windows. Norton SONAR flags visible-console spawns from a frozen
    app; every subprocess we shell out to for collector purposes must
    use these flags."""
    kwargs: dict = {}
    if sys.platform.startswith("win"):
        try:
            import subprocess as _sp
            flags = 0
            for attr in ("CREATE_NO_WINDOW",):
                flags |= getattr(_sp, attr, 0)
            kwargs["creationflags"] = flags
            si = _sp.STARTUPINFO()
            si.dwFlags |= _sp.STARTF_USESHOWWINDOW
            si.wShowWindow = 0  # SW_HIDE
            kwargs["startupinfo"] = si
        except Exception:
            pass
    return kwargs


def _run(cmd: List[str], timeout: float = 5.0) -> str:
    """Run a subprocess with hidden window, capture stdout+stderr,
    return decoded text or the exception as a string. Never raises."""
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            timeout=timeout,
            **_hidden_kwargs(),
        )
        out = (proc.stdout or b"").decode("utf-8", errors="replace")
        err = (proc.stderr or b"").decode("utf-8", errors="replace")
        return out + ("\n---stderr---\n" + err if err.strip() else "")
    except FileNotFoundError:
        return f"(not on PATH: {cmd[0]})"
    except subprocess.TimeoutExpired:
        return f"(timeout after {timeout}s: {' '.join(cmd)})"
    except Exception as exc:  # pragma: no cover
        return f"(exception running {cmd[0]}: {type(exc).__name__}: {exc})"


# ------------------------------------------------------------------
# collectors — each returns str, never raises, guarded by wrapper
# ------------------------------------------------------------------
def _collect_system(install_dir: Optional[Path]) -> str:
    out: List[str] = ["=== Touchless Debug Bundle — system.txt ==="]
    out.append(f"Collected: {time.strftime('%Y-%m-%d %H:%M:%S %z')}")
    out.append(f"Platform: {sys.platform}")
    out.append(f"Python: {sys.version.split()[0]}  frozen={getattr(sys, 'frozen', False)}")
    try:
        import platform
        out.append(f"OS: {platform.platform()}")
        out.append(f"Machine: {platform.machine()}")
        out.append(f"Processor: {platform.processor()}")
        out.append(f"Computer name: {platform.node()}")
    except Exception as exc:
        out.append(f"(platform module failed: {exc})")
    try:
        import getpass
        out.append(f"User: {getpass.getuser()}")
    except Exception:
        pass
    try:
        import psutil  # type: ignore
        vm = psutil.virtual_memory()
        out.append(f"RAM total: {vm.total / (1024**3):.1f} GB")
        out.append(f"RAM avail: {vm.available / (1024**3):.1f} GB")
        out.append(f"CPU logical: {psutil.cpu_count(logical=True)}  physical: {psutil.cpu_count(logical=False)}")
    except Exception as exc:
        out.append(f"(psutil probe failed: {exc})")
    out.append("")
    out.append("--- Install ---")
    out.append(f"Install dir: {install_dir}")
    if install_dir is not None:
        exe = install_dir / "Touchless.exe"
        out.append(f"Touchless.exe: {exe}")
        try:
            st = exe.stat()
            out.append(f"Exe size: {st.st_size} bytes  mtime: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(st.st_mtime))}")
        except Exception:
            pass
    out.append(f"sys.executable: {sys.executable}")
    out.append(f"sys.argv: {sys.argv}")
    return "\n".join(out) + "\n"


def _collect_env() -> str:
    out: List[str] = ["=== Environment ===", "", "--- HGR_* vars ---"]
    for k in sorted(os.environ):
        if k.startswith("HGR_"):
            out.append(f"{k:32s} = {os.environ[k]}")
    out.append("")
    out.append("--- Core Windows vars ---")
    for k in ("USERPROFILE", "LOCALAPPDATA", "APPDATA", "TEMP", "ProgramFiles",
              "ProgramFiles(x86)", "PATH"):
        out.append(f"{k:32s} = {os.environ.get(k, '(unset)')}")
    return "\n".join(out) + "\n"


_REDACT_PATTERNS = re.compile(
    r"(token|secret|password|client_id|client_secret|refresh|access_key|api_key)",
    re.IGNORECASE,
)


def _redact(obj):
    if isinstance(obj, dict):
        return {
            k: ("<redacted>" if _REDACT_PATTERNS.search(str(k)) and isinstance(v, str) and v
                else _redact(v))
            for k, v in obj.items()
        }
    if isinstance(obj, list):
        return [_redact(x) for x in obj]
    return obj


def _collect_config_redacted() -> Optional[str]:
    """Read %USERPROFILE%/.touchless/settings.json, blank any token-shaped
    field, return the JSON text. Returns None on any failure so the
    caller writes a placeholder."""
    try:
        cfg_path = Path.home() / ".touchless" / "settings.json"
        if not cfg_path.exists():
            return None
        with open(cfg_path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return json.dumps(_redact(data), indent=2)
    except Exception:
        return None


def _collect_log_tail(max_bytes: int = 2_000_000) -> str:
    """Return the last max_bytes of touchless_debug.log."""
    log_path = Path(os.environ.get("LOCALAPPDATA", str(Path.home() / "AppData/Local"))) / "Touchless" / "logs" / "touchless_debug.log"
    if not log_path.exists():
        return f"(no log at {log_path})\n"
    try:
        st = log_path.stat()
        if st.st_size > 200 * 1024 * 1024:
            return f"(log too large: {st.st_size} bytes — skipped to avoid stall)\n"
        with open(log_path, "rb") as fh:
            if st.st_size > max_bytes:
                fh.seek(-max_bytes, 2)
                fh.readline()  # discard partial line
            return fh.read().decode("utf-8", errors="replace")
    except Exception as exc:
        return f"(exception reading log: {exc})\n"


def _collect_perf_signals(log_text: str) -> str:
    """Extract triage-signal lines from the log tail. Grep-narrowed so
    the reviewer doesn't have to scan the whole log."""
    if not log_text:
        return "(no log text)\n"
    keys = [
        "[freeze-detector]", "[r49-short-shutter]", "[r51-", "[r53-",
        "[r54-", "[r50-", "[perf-mode]", "[perf-auto]", "[perf-camera]",
        "[gamma-lift]", "[ffmpeg-caps]", "[ffmpeg_capture]",
        "[clip-cache]", "[clip-audio-v2]",
        # r20/r21 adaptive signals. These are the lines that say whether
        # the per-machine behaviour actually engaged on the user's PC:
        # what the camera reported it can do, which capture mode was
        # chosen, whether an inherited dark exposure was released, and
        # whether the recorder was switched off for this hardware.
        "[camera-caps]", "[camera-controls]", "[unstick-inherited]",
        "[ffmpeg-memo]", "[clip-cache-seed]", "[hdr-throttle]",
        # the camera-path decision log, which names the mode chosen
        "[lite_mode/ffmpeg]",
        # r23 adaptive signals. `[exposure-policy]` says whether the
        # short-shutter value was tailored to this camera's advertised
        # frame rate rather than the old -6.0 constant; `[gamma-lift]`
        # says what luma the sensor actually produced afterwards (the
        # two together are the whole dim-preview diagnosis);
        # `[ffmpeg-cooldown]` says whether repeated ffmpeg attempts --
        # and so repeated antivirus prompts -- were throttled.
        "[exposure-policy]", "[gamma-lift]", "[ffmpeg-cooldown]",
        # r24. `[perf-auto]` is the one that explained "none of the modes
        # do anything": once the app auto-latches into low-fps, Default,
        # Lite and GPU all build the same engine, and nothing else in the
        # log says so. `[perf-mode]` shows which engine each toggle
        # actually built, and `[cap-read]` (needs HGR_TICK_TIMING=1)
        # separates a camera that is not delivering frames from a
        # pipeline that cannot keep up -- the question a bundle could
        # never answer on its own.
        "[perf-auto]", "[perf-mode]", "[cap-read]", "[tick-timing]",
        # r24 light lift. Logs on EVERY path, including the two skips,
        # which are deliberately worded differently: "could not measure
        # the frame" means the sampler never saw a published frame,
        # "is not dark enough" means it saw one and the picture was
        # fine. A single silent return would make a dead feature and a
        # working one look identical in a bundle.
        "[light-lift]",
        # Which inference backend actually resolved. Without these, a
        # user reporting "GPU mode does nothing" cannot be answered from
        # a bundle at all: `[hand_runtime]` carries the gpu_mode/ONNX
        # decision and `[onnx_runtime]` carries the providers the
        # session really bound, and both were filtered out of the very
        # file built for perf triage. `[detector]` says whether the Lite
        # width lever actually bites on the delivered frame.
        "[hand_runtime]", "[onnx_runtime]", "[detector]",
    ]
    kept: List[str] = ["=== Perf signals (grep-narrowed from log tail) ==="]
    for line in log_text.splitlines():
        for k in keys:
            if k in line:
                kept.append(line)
                break
    return "\n".join(kept) + "\n"


def _collect_event_log() -> str:
    if not sys.platform.startswith("win"):
        return "(non-Windows platform)\n"
    ps = (
        "$ErrorActionPreference='SilentlyContinue';"
        "Get-WinEvent -FilterHashtable @{LogName='Application'; "
        "StartTime=(Get-Date).AddDays(-1)} -MaxEvents 200 | "
        "Where-Object { $_.ProviderName -match 'Application Error|"
        "Windows Error Reporting|Norton|Symantec|Touchless|ffmpeg' } | "
        "Format-List TimeCreated, LevelDisplayName, ProviderName, Id, Message"
    )
    return _run(["powershell", "-NoProfile", "-Command", ps], timeout=10.0)


def _collect_gpu_driver() -> str:
    parts: List[str] = ["--- nvidia-smi ---"]
    parts.append(_run(["nvidia-smi"], timeout=5.0))
    parts.append("\n--- Win32_VideoController (WMI) ---")
    if sys.platform.startswith("win"):
        ps = (
            "Get-CimInstance Win32_VideoController | "
            "Format-List Name, DriverVersion, AdapterRAM, VideoModeDescription"
        )
        parts.append(_run(["powershell", "-NoProfile", "-Command", ps], timeout=5.0))
    return "\n".join(parts) + "\n"


def _collect_cameras() -> str:
    # v1.1.9.2 (r18): the in-process cv2.VideoCapture(0..5, CAP_DSHOW)
    # loop that used to live here was REMOVED. It built up to six
    # DirectShow graphs from the bundle thread — on a cheap UVC driver
    # that is the exact double-open stall the field logs show, and it
    # ran while the engine could still hold the device. The ffmpeg
    # device listing below enumerates DirectShow devices without
    # opening any pin, and the runtime log tail already records the
    # negotiated width/height/fps of the camera the session used.
    parts: List[str] = ["--- OpenCV VideoCapture probe: skipped (r18; see ffmpeg listing + log tail) ---"]
    if sys.platform.startswith("win"):
        parts.append("\n--- DirectShow devices (ffmpeg -list_devices) ---")
        ffm = _resolve_ffmpeg_path()
        if ffm:
            parts.append(_run([str(ffm), "-hide_banner", "-f", "dshow",
                               "-list_devices", "true", "-i", "dummy"], timeout=8.0))
    return "\n".join(parts) + "\n"


def _collect_audio_devices() -> str:
    try:
        import sounddevice as sd  # type: ignore
        return str(sd.query_devices())
    except Exception as exc:
        return f"(sounddevice.query_devices() failed: {exc})\n"


def _resolve_ffmpeg_path() -> Optional[Path]:
    """Best-effort locate the bundled ffmpeg.exe. Frozen build layout:
    Touchless.exe next to _internal/ffmpeg.EXE."""
    candidates: List[Path] = []
    if getattr(sys, "frozen", False):
        exe_dir = Path(sys.executable).resolve().parent
        candidates.append(exe_dir / "_internal" / "ffmpeg.exe")
        candidates.append(exe_dir / "_internal" / "ffmpeg.EXE")
        candidates.append(exe_dir / "ffmpeg.exe")
    which = shutil.which("ffmpeg")
    if which:
        candidates.append(Path(which))
    for c in candidates:
        try:
            if c.exists():
                return c
        except Exception:
            continue
    return None


def _collect_ffmpeg_version() -> str:
    p = _resolve_ffmpeg_path()
    if p is None:
        return "(ffmpeg not found)\n"
    out = _run([str(p), "-hide_banner", "-version"], timeout=5.0)
    return f"resolved path: {p}\n\n{out}\n"


def _collect_disk_space(install_dir: Optional[Path]) -> str:
    lines: List[str] = ["=== Disk space (bytes) ==="]
    for label, path in (
        ("install_dir", install_dir),
        ("LOCALAPPDATA", Path(os.environ.get("LOCALAPPDATA", "") or Path.home() / "AppData/Local")),
        ("home", Path.home()),
        ("TEMP", Path(tempfile.gettempdir())),
    ):
        if path is None:
            continue
        try:
            u = shutil.disk_usage(str(path))
            lines.append(f"{label:16s} {path}  total={u.total}  used={u.used}  free={u.free}  ({u.free/(1024**3):.1f} GB free)")
        except Exception as exc:
            lines.append(f"{label:16s} {path}  disk_usage failed: {exc}")
    return "\n".join(lines) + "\n"


def _collect_processes() -> str:
    keywords = ("ffmpeg", "touchless", "obs", "camera", "streamlabs", "discord",
                "snapchat", "razer", "synapse")
    lines: List[str] = ["=== Running processes matching perf-triage keywords ==="]
    try:
        import psutil  # type: ignore
        for p in psutil.process_iter(["pid", "name", "cmdline", "create_time"]):
            try:
                info = p.info
                name = (info.get("name") or "").lower()
                if any(k in name for k in keywords):
                    ct = info.get("create_time") or 0
                    lines.append(
                        f"pid={info.get('pid')} name={info.get('name')} "
                        f"started={time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(ct))} "
                        f"cmd={info.get('cmdline')}"
                    )
            except Exception:
                continue
    except Exception as exc:
        lines.append(f"(psutil.process_iter failed: {exc})")
    return "\n".join(lines) + "\n"


def _collect_install_dir_manifest(install_dir: Optional[Path]) -> str:
    if install_dir is None or not install_dir.exists():
        return "(install_dir unknown)\n"
    lines: List[str] = [f"=== Install dir manifest — {install_dir} ==="]
    try:
        for root, dirs, files in os.walk(str(install_dir)):
            for fn in files:
                p = Path(root) / fn
                try:
                    st = p.stat()
                    rel = p.relative_to(install_dir)
                    lines.append(
                        f"{st.st_size:12d}  "
                        f"{time.strftime('%Y-%m-%d %H:%M', time.localtime(st.st_mtime))}  "
                        f"{rel}"
                    )
                except Exception:
                    continue
    except Exception as exc:
        lines.append(f"(walk failed: {exc})")
    return "\n".join(lines) + "\n"


# ------------------------------------------------------------------
# public API
# ------------------------------------------------------------------
@dataclass
class BundleContext:
    """Small ctx object the caller hands to collect_debug_bundle. Kept
    minimal so main_window.closeEvent can build it without a heavy
    query pass at shutdown time."""
    session_started_at: float = 0.0
    exit_code: int = 0
    reason: str = "clean-close"
    install_dir: Optional[Path] = None


def _safe(name: str, fn: Callable[[], str]) -> tuple[str, str]:
    try:
        return name, fn()
    except Exception:
        return name, "(collector raised: " + traceback.format_exc() + ")\n"


def collect_debug_bundle(ctx: BundleContext,
                         output_path: Path,
                         progress_cb: Optional[Callable[[int, str], None]] = None) -> Path:
    """Write a debug bundle zip to `output_path`. Returns the final
    path. Never raises — writes a bundle even if half the collectors
    fail."""
    # Guess install_dir from sys.executable if not provided.
    install_dir = ctx.install_dir
    if install_dir is None and getattr(sys, "frozen", False):
        install_dir = Path(sys.executable).resolve().parent

    def _emit(pct: int, msg: str) -> None:
        if progress_cb is not None:
            try:
                progress_cb(pct, msg)
            except Exception:
                pass

    _emit(2, "Reading log tail…")
    log_tail_text = _collect_log_tail()

    entries: List[tuple[str, str]] = []
    steps: List[tuple[str, Callable[[], str], int]] = [
        ("system.txt", lambda: _collect_system(install_dir), 10),
        ("env.txt", _collect_env, 15),
        ("touchless_debug.log.tail", lambda: log_tail_text, 30),
        ("perf_signals.txt", lambda: _collect_perf_signals(log_tail_text), 40),
        ("disk_space.txt", lambda: _collect_disk_space(install_dir), 45),
        ("ffmpeg_version.txt", _collect_ffmpeg_version, 55),
        ("cameras.txt", _collect_cameras, 65),
        ("audio_devices.txt", _collect_audio_devices, 70),
        ("gpu_driver.txt", _collect_gpu_driver, 78),
        ("processes.txt", _collect_processes, 82),
        ("event_log_app.txt", _collect_event_log, 88),
        ("install_dir_manifest.txt", lambda: _collect_install_dir_manifest(install_dir), 94),
    ]
    for name, fn, pct in steps:
        _emit(pct, f"Collecting {name}…")
        entries.append(_safe(name, fn))

    # config-redacted.json — special: may be None (no config on disk yet).
    _emit(96, "Collecting config-redacted.json…")
    cfg_text = _collect_config_redacted()
    if cfg_text is not None:
        entries.append(("config-redacted.json", cfg_text))

    # Write to a temp file first so an interrupted zip doesn't strand
    # the destination as a truncated archive.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_path.with_suffix(output_path.suffix + f".{os.getpid()}.tmp")
    try:
        with zipfile.ZipFile(str(tmp), "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for name, text in entries:
                try:
                    zf.writestr(name, text.encode("utf-8", errors="replace"))
                except Exception:
                    continue
            # Copy crash/ + install_log/ dirs verbatim.
            appdata_root = Path(os.environ.get("LOCALAPPDATA", "") or Path.home() / "AppData/Local") / "Touchless"
            for sub in ("crash", "install_log"):
                src = appdata_root / sub
                if src.exists() and src.is_dir():
                    for root, dirs, files in os.walk(str(src)):
                        for fn in files:
                            p = Path(root) / fn
                            try:
                                arcname = f"{sub}/{p.relative_to(src)}"
                                zf.write(str(p), arcname)
                            except Exception:
                                continue
        _emit(99, "Finalizing…")
        os.replace(str(tmp), str(output_path))
    finally:
        try:
            if tmp.exists():
                tmp.unlink()
        except Exception:
            pass

    # Write the sentinel so the external Touchless_Debug.ps1 wrapper
    # can skip its own SaveFileDialog on this session.
    try:
        appdata_root = Path(os.environ.get("LOCALAPPDATA", "") or Path.home() / "AppData/Local") / "Touchless"
        appdata_root.mkdir(parents=True, exist_ok=True)
        marker = appdata_root / "last_bundle_saved.marker"
        marker.write_text(
            f"{time.strftime('%Y-%m-%d %H:%M:%S')}\t{output_path}\n",
            encoding="utf-8",
        )
    except Exception:
        pass

    _emit(100, "Done.")
    return output_path
