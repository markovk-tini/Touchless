# Touchless on macOS — Build, Run & Test

> Companion to [docs/MACOS_PORT.md](MACOS_PORT.md) (the full Windows→macOS
> compatibility audit + porting plan). This file is the practical "how do I run
> it on my Mac" guide for the **base build**.
>
> **Everything here is a NEW, SEPARATE pipeline.** None of the Windows build
> files are modified: `builder/windows/*`, `installers/windows/hgr_app.iss`,
> `signing/sign-file.bat`, `requirements.txt`, and `run_test.py` are untouched.

## TL;DR

```bash
# On the Mac, from the repo root:
./run_mac.sh --smoke     # headless import/capability check (no GUI, no camera prompt)
./run_mac.sh             # set up .venv-mac, install deps, launch the app from source
```

To build a distributable app bundle:

```bash
./builder/macos/build_mac.sh          # dev build -> dist/Touchless.app (ad-hoc signed)
./builder/macos/build_mac.sh --pkg    # also build dist/Touchless-<ver>.pkg
```

## What's in the base build (Phase 0)

The base build is everything the audit marked **✅ works-as-is** plus the new
packaging/run scaffolding. Concretely, it should:

- **Import and launch** on macOS. The app's Windows API usage is all runtime
  `ctypes.windll` calls inside platform-guarded functions — there are no
  module-level `import win32…` / `pycaw` / `comtypes`, so the startup chain
  imports cleanly on macOS. (`./run_mac.sh --smoke` verifies this.)
- Show the **PySide6 UI** (renders via Cocoa/AppKit).
- Open the **camera** via the existing AVFoundation `Darwin` branch in
  `camera_utils` (prompts for the Camera permission on first open).
- Run the **gesture pipeline** — MediaPipe/ONNX tracking, classifiers, feature
  extraction, smoothing, custom-gesture recording — all pure-compute, identical
  to Windows.

## What is NOT in the base build yet (later, Mac-hardware-gated phases)

These need real Quartz/Accessibility/AppKit implementations and on-device
testing (see the phased plan in [MACOS_PORT.md](MACOS_PORT.md)):

- **Cursor & keyboard control** (gesture→cursor, dictation text injection) →
  Quartz `CGEvent*`, gated on the **Accessibility** permission.
- **Click-through overlay** final wiring (`setIgnoresMouseEvents_`, per-overlay
  window level). The pyobjc `Darwin` branch already exists in `native_overlay.py`.
- **System volume / media keys** → CoreAudio + `NX_KEYTYPE_*` (per-app volume is
  **not available** on macOS — a documented difference).
- **App controllers** (Chrome/Spotify/Discord/Office) → AppleScript/Automation.
- **Voice with Metal** → `whisper.cpp`/`llama.cpp` rebuilt with `-DGGML_METAL=ON`
  (`builder/macos/_build_*_metal.sh`). Base build falls back to faster-whisper (CPU).
- **Iris / live_api** → AXUIElement, ScreenCaptureKit, Keychain.
- **First-run permission onboarding** wizard, code-signing + notarization.

## Files (all new)

| File | Purpose |
|---|---|
| `requirements_mac.txt` | macOS deps (pyobjc, plain onnxruntime; no pywin32/pycaw/comtypes/DirectML) |
| `run_mac.sh` | Dev launcher: makes `.venv-mac`, installs deps, runs from source (`--smoke` for headless) |
| `scripts/mac_smoke_test.py` | Headless import/capability smoke test |
| `src/hgr/platform_compat/` | Additive cross-platform layer: `dirs` (paths) + `capabilities` (feature flags) |
| `builder/macos/hgr_app_mac.spec` | PyInstaller `.app` spec (arm64, Info.plist TCC strings) |
| `builder/macos/build_mac.sh` | Build orchestrator (venv → pyinstaller → codesign → pkg/notarize) |
| `builder/macos/_build_whisper_metal.sh` / `_build_llama_metal.sh` | Metal native rebuilds (voice phase) |
| `signing/macos/entitlements.plist` | Hardened-runtime entitlements |
| `installers/macos/build_pkg.sh` + `distribution.xml` | `.pkg` installer (pkgbuild/productbuild) |
| `installers/macos/build_dmg.sh` | drag-to-Applications `.dmg` |

## Signing & distribution notes

- **Dev builds** are **ad-hoc signed** (`codesign -s -`) so the app launches
  locally. ⚠️ TCC permission grants reset on every ad-hoc rebuild — fine for
  launch testing, not for distribution.
- **Release builds** need an **Apple Developer ID** (Developer Program, $99/yr):
  set `DEVELOPER_ID_APP` / `DEVELOPER_ID_INSTALLER` and the `AC_API_*`
  notarization creds, then `build_mac.sh --pkg --notarize`.
- The **Mac App Store is off the table** for the full app (App Sandbox forbids
  the global Accessibility / CGEvent input control). Channel is Developer-ID
  direct download (`.pkg`/`.dmg`).

## Known first-run gotchas

- **mediapipe wheel** — `requirements_mac.txt` pins `mediapipe==0.10.21` to match
  Windows. If pip can't find an arm64 wheel for that exact pin on your macOS,
  bump to the newest `0.10.x` (same hand model → identical accuracy) and tell us.
- **Camera/mic prompts** — first access triggers a system permission dialog; the
  Info.plist usage strings (in the spec) are mandatory or the app crashes on access.
- **`from ctypes import wintypes`** appears at module top-level in a few files;
  it's pure-stdlib type aliases and imports fine on macOS, but the smoke test
  confirms it on real hardware.
