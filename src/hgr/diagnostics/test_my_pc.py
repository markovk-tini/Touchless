"""Test-My-PC library — hardware probes + per-mode benchmark used by
the CLI at tools/test_my_pc.py.

Every function here is pure — no prints. Progress and cancellation
are surfaced via optional callbacks so a future in-app dialog could
drive a progress bar without blocking the GUI thread if we ever
promote this from a debug-only tool to a shipped feature.
"""
from __future__ import annotations

import os
import platform
import statistics
import subprocess
import sys
import time
from typing import Callable, List, Optional


# --------------------------------------------------------------- mode presets

MODES: List[dict] = [
    {
        "label": "Default",
        "model_complexity": 1,
        "max_process_width": 960,
        "prefer_gpu": False,
    },
    {
        "label": "Lite",
        "model_complexity": 1,
        "max_process_width": 640,
        "prefer_gpu": False,
    },
    {
        "label": "GPU",
        "model_complexity": 1,
        "max_process_width": 960,
        "prefer_gpu": True,
    },
]


# ---------------------------------------------------------------- system info


def _sub_kwargs() -> dict:
    """Wrap the runtime `hidden_subprocess_kwargs` helper; if it can't
    import (running from a partial checkout), fall back to plain kwargs
    so the probes still work at the cost of a brief console flash."""
    try:
        from hgr.utils.subprocess_utils import hidden_subprocess_kwargs
        return hidden_subprocess_kwargs()
    except Exception:
        return {}


def cpu_info() -> dict:
    info: dict = {"name": platform.processor() or "unknown", "cores": os.cpu_count() or 0}
    try:
        import psutil  # type: ignore

        info["logical_cores"] = psutil.cpu_count(logical=True)
        info["physical_cores"] = psutil.cpu_count(logical=False)
        freq = psutil.cpu_freq()
        if freq is not None:
            info["current_mhz"] = int(freq.current)
            info["max_mhz"] = int(freq.max) if freq.max else None
    except Exception:
        pass
    if platform.system() == "Windows":
        # PowerShell first (works on Win11 24H2+ where wmic was removed),
        # then wmic as a fallback for older machines.
        for cmd in (
            ["powershell", "-NoProfile", "-Command",
             "Get-CimInstance Win32_Processor | Select-Object -ExpandProperty Name"],
            ["wmic", "cpu", "get", "Name"],
        ):
            try:
                out = subprocess.check_output(
                    cmd,
                    stderr=subprocess.DEVNULL,
                    timeout=5.0,
                    **_sub_kwargs(),
                ).decode(errors="ignore")
                lines = [ln.strip() for ln in out.splitlines()
                         if ln.strip() and "Name" not in ln]
                if lines:
                    info["name"] = lines[0]
                    break
            except Exception:
                continue
    return info


def ram_info() -> dict:
    try:
        import psutil  # type: ignore

        mem = psutil.virtual_memory()
        return {
            "total_gb": round(mem.total / (1024**3), 1),
            "available_gb": round(mem.available / (1024**3), 1),
        }
    except Exception:
        return {}


def gpu_info() -> dict:
    info: dict = {"name": "unknown", "vram_mb": None, "directml_available": None}
    if platform.system() == "Windows":
        # PowerShell first, wmic fallback (see cpu_info comment).
        parsed = False
        for cmd, is_ps in (
            (["powershell", "-NoProfile", "-Command",
              "Get-CimInstance Win32_VideoController | "
              "Select-Object Name,AdapterRAM | Format-Table -HideTableHeaders"],
             True),
            (["wmic", "path", "win32_VideoController", "get", "Name,AdapterRAM"],
             False),
        ):
            try:
                out = subprocess.check_output(
                    cmd,
                    stderr=subprocess.DEVNULL,
                    timeout=5.0,
                    **_sub_kwargs(),
                ).decode(errors="ignore")
                lines = [ln.strip() for ln in out.splitlines()
                         if ln.strip() and "AdapterRAM" not in ln]
                if not lines:
                    continue
                first = lines[0]
                # Both wmic and the PS Format-Table above put "<name> <vram_bytes>"
                # on one line. Split off the trailing digit run as the VRAM.
                parts = first.rsplit(None, 1)
                if len(parts) == 2 and parts[1].isdigit():
                    info["vram_mb"] = int(parts[1]) // (1024**2)
                    info["name"] = parts[0].strip()
                elif not is_ps:
                    # wmic historically prints "<vram> <name>" (opposite order)
                    parts2 = first.split(None, 1)
                    if len(parts2) == 2 and parts2[0].isdigit():
                        info["vram_mb"] = int(parts2[0]) // (1024**2)
                        info["name"] = parts2[1].strip()
                    else:
                        info["name"] = first
                else:
                    info["name"] = first
                parsed = True
                break
            except Exception:
                continue
        if not parsed:
            pass
    # DirectML capability via onnxruntime provider list.
    try:
        import onnxruntime as ort  # type: ignore

        providers = ort.get_available_providers()
        info["directml_available"] = "DmlExecutionProvider" in providers
        info["onnxruntime_providers"] = list(providers)
    except Exception:
        info["directml_available"] = False
    return info


def camera_info() -> dict:
    """Camera list via the app's own enumerator when available, so the
    frozen build doesn't need ffmpeg on PATH."""
    devices: List[str] = []
    try:
        from hgr.app.camera.camera_utils import list_available_cameras
        for c in list_available_cameras() or []:
            # CameraInfo exposes display_name; the older "name"/
            # "device_name" spellings never existed, so this list
            # came back empty on every machine. Keep the legacy
            # getattrs last as a tolerance for other shapes.
            name = (
                getattr(c, "display_name", None)
                or getattr(c, "name", None)
                or getattr(c, "device_name", None)
            )
            if name and name not in devices:
                devices.append(name)
    except Exception:
        pass
    return {"devices": devices}


# --------------------------------------------- cheap hardware capability
#
# These two helpers exist so the app can adapt its defaults to the
# machine WITHOUT paying for probe_system(): that costs two PowerShell
# Get-CimInstance calls plus a DirectShow camera enumeration, about 4 s
# warm and 9.6 s cold, and it spawns processes that consumer antivirus
# then scans. Everything below reads the registry only.


#: Name fragments that mark an adapter as integrated rather than an
#: add-in card. Matched case-insensitively against DriverDesc.
INTEGRATED_ADAPTER_HINTS = (
    "intel(r) uhd", "intel(r) hd graphics", "intel(r) iris",
    "intel uhd", "intel hd graphics", "intel iris",
    "radeon(tm) graphics", "radeon(tm) vega", "amd radeon(tm) r",
    "microsoft basic display", "microsoft remote display",
    "parsec virtual", "citrix indirect", "virtual display",
)


def _looks_integrated(driver_desc: str) -> bool:
    name = str(driver_desc or "").strip().lower()
    if not name:
        return False
    return any(hint in name for hint in INTEGRATED_ADAPTER_HINTS)


def discrete_gpu_vram_mb() -> int:
    """Largest adapter VRAM in MB, read from the display-class registry.

    Uses the 64-bit `HardwareInformation.qwMemorySize` value, so it is
    free of the `Win32_VideoController.AdapterRAM` 32-bit ceiling that
    reports every card of 4 GB or more as exactly 4095 MB. Taking the
    maximum picks the discrete GPU over an integrated one regardless of
    enumeration order.

    Returns 0 when nothing could be read. Callers MUST treat 0 as
    "unknown" and fail open.
    """
    if not sys.platform.startswith("win"):
        return 0
    try:
        import winreg
    except Exception:
        return 0
    base = (r"SYSTEM\CurrentControlSet\Control\Class"
            r"\{4d36e968-e325-11ce-bfc1-08002be10318}")
    discrete = 0
    any_adapter = 0
    try:
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, base) as root:
            i = 0
            while True:
                try:
                    sub = winreg.EnumKey(root, i)
                except OSError:
                    break
                i += 1
                if not sub.isdigit():
                    continue
                desc = ""
                try:
                    with winreg.OpenKey(root, sub) as key:
                        raw = winreg.QueryValueEx(
                            key, "HardwareInformation.qwMemorySize"
                        )[0]
                        try:
                            desc = str(winreg.QueryValueEx(key, "DriverDesc")[0])
                        except OSError:
                            desc = ""
                except OSError:
                    continue
                try:
                    mb = int(raw) // (1024 * 1024)
                except Exception:
                    continue
                if mb <= 0:
                    continue
                if mb > any_adapter:
                    any_adapter = mb
                if not _looks_integrated(desc) and mb > discrete:
                    discrete = mb
    except Exception:
        return 0
    # Prefer a real add-in card. Taking a plain maximum was wrong: an
    # integrated GPU can report a large shared-memory figure, and on a
    # machine with a weak discrete card plus an iGPU that would report
    # the wrong number and hide exactly the hardware we care about.
    return discrete or any_adapter


#: Desktop area above which the always-on clip recorder starts to cost
#: real CPU: gdigrab hands ffmpeg a full BGRA frame of this size at
#: 20 fps and swscale then has to rescale it.
CLIP_CACHE_HEAVY_PIXELS = 3840 * 1200
#: VRAM at or below which a GPU is treated as unable to carry that load
#: alongside the gesture pipeline.
CLIP_CACHE_WEAK_VRAM_MB = 3072


def should_disable_clip_cache(desktop_pixels: int, vram_mb: int) -> bool:
    """Should the always-on clip recorder be OFF by default here?

    True only when the desktop is large AND the GPU is small. Both
    halves are required: a big desktop on a strong GPU is fine, and a
    weak GPU driving a small desktop is fine too. An unknown VRAM
    reading (0) always fails open, preserving the historical default.
    """
    try:
        px = int(desktop_pixels)
        vram = int(vram_mb)
    except Exception:
        return False
    if vram <= 0:
        return False
    return px > CLIP_CACHE_HEAVY_PIXELS and vram <= CLIP_CACHE_WEAK_VRAM_MB


def probe_system() -> dict:
    return {
        "os": f"{platform.system()} {platform.release()}",
        "os_version": platform.version(),
        "cpu": cpu_info(),
        "ram": ram_info(),
        "gpu": gpu_info(),
        "camera": camera_info(),
    }


# ------------------------------------------------------------ mode benchmark


def make_synthetic_frame(width: int = 640, height: int = 480):
    import numpy as np  # type: ignore

    rng = np.random.default_rng(42)
    frame = rng.integers(60, 200, (height, width, 3), dtype=np.uint8)
    ch, cw = height // 2, width // 2
    frame[ch - 40 : ch + 40, cw - 40 : cw + 40] = 220
    return frame


ProgressCb = Optional[Callable[[str, int, int], None]]     # (phase, done, total)
CancelCb = Optional[Callable[[], bool]]                    # returns True to abort


def benchmark_mode(
    label: str,
    *,
    model_complexity: int,
    max_process_width: int,
    prefer_gpu: bool,
    cap_width: int = 640,
    cap_height: int = 480,
    warmup: int = 60,
    measure: int = 300,
    progress_cb: ProgressCb = None,
    cancel_cb: CancelCb = None,
) -> dict:
    from hgr.gesture.tracking.detector import HandDetector

    if progress_cb is not None:
        progress_cb("build", 0, 1)
    t0 = time.perf_counter()
    detector = HandDetector(
        model_complexity=model_complexity,
        max_process_width=max_process_width,
        prefer_gpu=prefer_gpu,
    )
    build_ms = (time.perf_counter() - t0) * 1000.0
    backend = (getattr(detector, "runtime_backend", None)
               or getattr(detector, "backend", None) or "unknown")
    if progress_cb is not None:
        progress_cb("build", 1, 1)

    frame = make_synthetic_frame(cap_width, cap_height)
    times_ms: List[float] = []

    for i in range(warmup):
        if cancel_cb is not None and cancel_cb():
            return {"label": label, "error": "cancelled",
                    "build_ms": round(build_ms, 1), "backend": str(backend)}
        try:
            detector.process(frame)
        except Exception as exc:
            return {
                "label": label,
                "error": f"process failed during warmup: {exc!r}",
                "build_ms": round(build_ms, 1),
                "backend": str(backend),
            }
        if progress_cb is not None and (i + 1) % 10 == 0:
            progress_cb("warmup", i + 1, warmup)
    if progress_cb is not None:
        progress_cb("warmup", warmup, warmup)

    for i in range(measure):
        if cancel_cb is not None and cancel_cb():
            return {"label": label, "error": "cancelled",
                    "build_ms": round(build_ms, 1), "backend": str(backend),
                    "samples_so_far": len(times_ms)}
        t = time.perf_counter()
        try:
            detector.process(frame)
        except Exception as exc:
            return {
                "label": label,
                "error": f"process failed mid-measure: {exc!r}",
                "build_ms": round(build_ms, 1),
                "backend": str(backend),
                "samples_so_far": len(times_ms),
            }
        times_ms.append((time.perf_counter() - t) * 1000.0)
        if progress_cb is not None and (i + 1) % 15 == 0:
            progress_cb("measure", i + 1, measure)
    if progress_cb is not None:
        progress_cb("measure", measure, measure)

    times_ms.sort()
    median = statistics.median(times_ms)
    p95 = times_ms[int(len(times_ms) * 0.95)]
    p99 = times_ms[int(len(times_ms) * 0.99)]
    cpu_fps_ceiling = min(60.0, 1000.0 / max(median, 0.1))

    return {
        "label": label,
        "backend": str(backend),
        "build_ms": round(build_ms, 1),
        "median_ms": round(median, 2),
        "p95_ms": round(p95, 2),
        "p99_ms": round(p99, 2),
        "fps_ceiling_at_60cam": round(cpu_fps_ceiling, 1),
    }


def benchmark_all(
    *,
    warmup: int = 60,
    measure: int = 300,
    cap_width: int = 640,
    cap_height: int = 480,
    progress_cb: Optional[Callable[[int, str, int, int], None]] = None,
    cancel_cb: CancelCb = None,
) -> List[dict]:
    """Run all three modes. `progress_cb` receives
    (mode_idx, phase, done, total). Modes that error are still
    included in the result list with an "error" key."""
    results: List[dict] = []
    for idx, m in enumerate(MODES):
        def _cb(phase, done, total, _idx=idx):
            if progress_cb is not None:
                progress_cb(_idx, phase, done, total)
        try:
            r = benchmark_mode(
                m["label"],
                model_complexity=m["model_complexity"],
                max_process_width=m["max_process_width"],
                prefer_gpu=m["prefer_gpu"],
                cap_width=cap_width,
                cap_height=cap_height,
                warmup=warmup,
                measure=measure,
                progress_cb=_cb,
                cancel_cb=cancel_cb,
            )
        except Exception as exc:
            r = {"label": m["label"], "error": f"benchmark threw: {exc!r}"}
        results.append(r)
        if cancel_cb is not None and cancel_cb():
            break
    return results


# --------------------------------------------------------------- recommendation


def recommend(results: List[dict], gpu_info_dict: dict):
    """Return (name, reason, lines).

    `lines` is the per-mode summary suitable for display / print. `name`
    is one of "Default" / "Lite" / "GPU". The 1 ms hysteresis prevents
    flapping between two modes that finished within measurement noise
    of each other."""
    by_label = {r["label"]: r for r in results if "error" not in r}
    default = by_label.get("Default")
    lite = by_label.get("Lite")
    gpu = by_label.get("GPU")

    if not default or not lite:
        return ("Default", "unable to benchmark",
                ["  (unable to recommend — Default or Lite benchmark failed)"])

    lines: List[str] = []
    lines.append(
        f"  Default:  {default['median_ms']} ms/frame  →  ceiling ~{default['fps_ceiling_at_60cam']} fps"
    )
    lines.append(
        f"  Lite:     {lite['median_ms']} ms/frame  →  ceiling ~{lite['fps_ceiling_at_60cam']} fps"
    )
    if gpu:
        gpu_engaged = ("DirectML engaged" if gpu_info_dict.get("directml_available")
                       else "CPU fallback")
        lines.append(
            f"  GPU:      {gpu['median_ms']} ms/frame  →  ceiling ~{gpu['fps_ceiling_at_60cam']} fps  ({gpu_engaged})"
        )

    rec = "Default"
    reason = "safe baseline"
    if (gpu and gpu["median_ms"] + 1.0 < lite["median_ms"]
            and gpu_info_dict.get("directml_available")):
        rec = "GPU"
        reason = "DirectML available and fastest"
    elif lite["median_ms"] + 1.0 < default["median_ms"]:
        rec = "Lite"
        reason = "faster than Default, no GPU required"

    return (rec, reason, lines)
