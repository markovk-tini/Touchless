#!/usr/bin/env python3
"""Touchless — macOS base-build smoke test.

NEW test script for the macOS port (the Windows side uses run_test.py / pytest).
This runs headless-ish and answers ONE question: "does the macOS base build
import and stand up the for-sure-portable pieces without crashing?"

It deliberately does NOT exercise Windows-only controllers (cursor injection,
volume, UIA, etc.) — those are later, Mac-hardware-gated phases. It checks:

  1. Python / platform sanity
  2. The heavy native wheels import (numpy, cv2, mediapipe, onnxruntime, PySide6)
  3. onnxruntime exposes the CoreML execution provider (GPU path) — informational
  4. pyobjc / AppKit import (needed by native_overlay's macOS branch)
  5. The Touchless app import chain (hgr.app.main and friends) imports cleanly
  6. The gesture pipeline constructs
  7. Camera enumeration via the existing AVFoundation Darwin branch (informational)

Exit code 0 = base build is import-healthy. Non-zero = a hard failure that
would stop the app from launching. Run via:  ./run_mac.sh --smoke

Author: Konstantin Markov (macOS port)
"""
from __future__ import annotations

import importlib
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# (label, callable) -> returns (ok: bool, hard: bool, detail: str)
_RESULTS: list[tuple[str, bool, bool, str]] = []


def _check(label: str, hard: bool, fn) -> None:
    try:
        detail = fn() or "ok"
        _RESULTS.append((label, True, hard, str(detail)))
    except Exception as exc:  # noqa: BLE001 — a smoke test reports, never raises
        _RESULTS.append((label, False, hard, f"{type(exc).__name__}: {exc}"))


def _imp(modname: str) -> str:
    mod = importlib.import_module(modname)
    return getattr(mod, "__version__", "imported")


def _check_platform() -> str:
    if sys.platform != "darwin":
        raise RuntimeError(f"not macOS (sys.platform={sys.platform})")
    return f"macOS {platform.mac_ver()[0]} / {platform.machine()} / Python {platform.python_version()}"


def _check_onnx_coreml() -> str:
    import onnxruntime as ort

    providers = ort.get_available_providers()
    has_coreml = "CoreMLExecutionProvider" in providers
    return f"providers={providers} | CoreML={'YES' if has_coreml else 'NO (CPU only)'}"


def _check_pyobjc() -> str:
    import objc  # noqa: F401
    from AppKit import NSWindow  # noqa: F401

    return "pyobjc + AppKit ok"


def _check_app_import() -> str:
    # The real startup import chain (run_app.py -> hgr.app.main). If a Windows-only
    # module-level import slipped through, this is where it would blow up on macOS.
    importlib.import_module("hgr.config.app_config")
    importlib.import_module("hgr.utils.runtime_paths")
    importlib.import_module("hgr.app.single_instance")
    importlib.import_module("hgr.app.main")
    return "hgr.app.main import chain clean"


def _check_gesture_pipeline() -> str:
    # The pure-compute core that must work day-one on macOS.
    importlib.import_module("hgr.core.tracking.hand_tracker")
    importlib.import_module("hgr.gesture.recognition.static_recognizer")
    return "gesture pipeline modules import"


def _check_camera_enum() -> str:
    from hgr.app.camera import camera_utils

    # _backend_candidates() returns the OS-appropriate cv2 backend order; on
    # macOS it should resolve to AVFoundation. This does not open the camera
    # (no TCC prompt), it just confirms the Darwin branch is selected.
    candidates = camera_utils._backend_candidates()
    names = [camera_utils.backend_name(b) for b in candidates]
    return f"camera backend order: {names}"


def main() -> int:
    print("=" * 68)
    print(" Touchless — macOS base-build smoke test")
    print("=" * 68)

    _check("platform / python", True, _check_platform)
    _check("import numpy", True, lambda: _imp("numpy"))
    _check("import cv2 (opencv)", True, lambda: _imp("cv2"))
    _check("import mediapipe", True, lambda: _imp("mediapipe"))
    _check("import onnxruntime", True, lambda: _imp("onnxruntime"))
    _check("import PySide6", True, lambda: _imp("PySide6"))
    _check("import sounddevice", False, lambda: _imp("sounddevice"))
    _check("onnxruntime CoreML provider", False, _check_onnx_coreml)
    _check("pyobjc / AppKit", True, _check_pyobjc)
    _check("Touchless app import chain", True, _check_app_import)
    _check("gesture pipeline import", True, _check_gesture_pipeline)
    _check("camera enumeration (AVFoundation)", False, _check_camera_enum)

    print()
    hard_failures = 0
    for label, ok, hard, detail in _RESULTS:
        mark = "PASS" if ok else ("FAIL" if hard else "warn")
        if not ok and hard:
            hard_failures += 1
        tag = "" if (ok or hard) else "  (non-fatal)"
        print(f"  [{mark}] {label:<34} {detail}{tag}")

    print()
    if hard_failures:
        print(f"RESULT: {hard_failures} hard failure(s) — base build is NOT launch-ready yet.")
        return 1
    print("RESULT: base build is import-healthy. Try './run_mac.sh' to launch the UI.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
