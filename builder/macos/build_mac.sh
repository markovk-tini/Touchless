#!/usr/bin/env bash
# Touchless — macOS build orchestrator (the structural analog of
# builder/windows/build_windows.bat, entirely SEPARATE — no Windows file is
# touched). Produces dist/Touchless.app, signs it, and optionally builds a
# notarized .pkg.
#
# Quick start (dev build, ad-hoc signed, no notarization):
#     ./builder/macos/build_mac.sh
#
# Release build (requires an Apple Developer ID + notarization creds in env):
#     export DEVELOPER_ID_APP="Developer ID Application: Your Name (TEAMID)"
#     export DEVELOPER_ID_INSTALLER="Developer ID Installer: Your Name (TEAMID)"
#     export AC_API_KEY_ID=... AC_API_ISSUER_ID=... AC_API_KEY_P8=/path/to/AuthKey.p8
#     ./builder/macos/build_mac.sh --pkg --notarize
#
# Flags:
#     --pkg         also build a .pkg installer (installers/macos/build_pkg.sh)
#     --dmg         also build a drag-to-Applications .dmg
#     --notarize    submit the .pkg/.app to Apple notarization + staple
#     --no-install  skip pip install (reuse the existing .venv-mac)
#     --metal       build whisper/llama Metal binaries (voice phase; OFF by default)
#     --skip-metal  accepted no-op (Metal is already off by default)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

VENV="$ROOT/.venv-mac"
PYBIN="${PYTHON:-python3}"
DO_INSTALL=1
DO_PKG=0
DO_DMG=0
DO_NOTARIZE=0
DO_METAL=0   # base build skips Metal; voice/Iris-local phase turns it on

for arg in "$@"; do
  case "$arg" in
    --pkg) DO_PKG=1 ;;
    --dmg) DO_DMG=1 ;;
    --notarize) DO_NOTARIZE=1 ;;
    --no-install) DO_INSTALL=0 ;;
    --skip-metal) DO_METAL=0 ;;
    --metal) DO_METAL=1 ;;
    -h|--help) grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "build_mac.sh: unknown arg '$arg' (try --help)"; exit 2 ;;
  esac
done

[[ "$(uname -s)" == "Darwin" ]] || { echo "build_mac.sh must run on macOS."; exit 1; }

# ---- 1. venv + deps ---------------------------------------------------------
[[ -d "$VENV" ]] || "$PYBIN" -m venv "$VENV"
# shellcheck disable=SC1091
source "$VENV/bin/activate"
VPY="$VENV/bin/python"

# Preflight: an x86_64 (Rosetta) interpreter would silently pull x86_64 wheels
# and fail the arm64 lipo check deep into the PyInstaller run. Catch it now.
"$VPY" - <<'PY'
import platform, sys
if platform.machine() != "arm64":
    sys.exit(f"ERROR: interpreter is {platform.machine()}, not arm64. "
             "Use a native Apple Silicon python3 (not Rosetta).")
PY

if [[ "$DO_INSTALL" == "1" ]]; then
  "$VPY" -m pip install --upgrade pip wheel
  "$VPY" -m pip install -r "$ROOT/requirements_mac.txt"
  "$VPY" -m pip install "pyinstaller>=6.0"
fi

# ---- 2. optional Metal native rebuilds (whisper.cpp / llama.cpp) ------------
if [[ "$DO_METAL" == "1" ]]; then
  echo "[build_mac] building Metal native binaries ..."
  bash "$ROOT/builder/macos/_build_whisper_metal.sh" || echo "[build_mac] whisper Metal build failed (continuing; CPU fallback)"
  bash "$ROOT/builder/macos/_build_llama_metal.sh"   || echo "[build_mac] llama Metal build failed (continuing)"
fi

# ---- 3. .icns icon (generated from the 1024 PNG) ---------------------------
ICNS="$ROOT/assets/icons/touchless_icon.icns"
PNG="$ROOT/assets/icons/touchless_icon.png"
if [[ ! -f "$ICNS" && -f "$PNG" ]]; then
  echo "[build_mac] generating $ICNS from $PNG"
  ICONSET="$(mktemp -d)/icon.iconset"; mkdir -p "$ICONSET"
  for sz in 16 32 64 128 256 512; do
    sips -z $sz $sz "$PNG" --out "$ICONSET/icon_${sz}x${sz}.png" >/dev/null
    sips -z $((sz*2)) $((sz*2)) "$PNG" --out "$ICONSET/icon_${sz}x${sz}@2x.png" >/dev/null
  done
  iconutil -c icns "$ICONSET" -o "$ICNS" || echo "[build_mac] iconutil failed — building without a custom icon"
fi

# ---- 4. PyInstaller -> dist/Touchless.app ----------------------------------
echo "[build_mac] running PyInstaller ..."
rm -rf "$ROOT/build/Touchless" "$ROOT/dist/Touchless.app"
"$VPY" -m PyInstaller "$ROOT/builder/macos/hgr_app_mac.spec" --noconfirm --clean
APP="$ROOT/dist/Touchless.app"
[[ -d "$APP" ]] || { echo "ERROR: $APP was not produced."; exit 1; }

# ---- 5. codesign ------------------------------------------------------------
ENTITLEMENTS="$ROOT/signing/macos/entitlements.plist"
HELPER_ENTITLEMENTS="$ROOT/signing/macos/entitlements-helper.plist"
if [[ -n "${DEVELOPER_ID_APP:-}" ]]; then
  echo "[build_mac] codesigning with Developer ID (inside-out) ..."
  # Sign every nested Mach-O first (dylibs, .so, frameworks, helper .apps),
  # then the outer bundle. --deep is unreliable for notarization, so we walk.
  # (BSD find: -maxdepth/-mindepth must precede the other primaries.)
  find "$APP/Contents" \( -name "*.dylib" -o -name "*.so" \) -print0 |
    while IFS= read -r -d '' f; do
      codesign --force --options runtime --timestamp \
        --entitlements "$ENTITLEMENTS" -s "$DEVELOPER_ID_APP" "$f"
    done
  find "$APP/Contents" -maxdepth 6 -name "*.framework" -print0 |
    while IFS= read -r -d '' fw; do
      codesign --force --options runtime --timestamp -s "$DEVELOPER_ID_APP" "$fw"
    done
  # Nested helper apps (e.g. QtWebEngineProcess.app) get the MINIMAL helper
  # entitlements (inherit + jit), NOT the parent's camera/mic/apple-events set.
  find "$APP/Contents" -mindepth 1 -name "*.app" -print0 |
    while IFS= read -r -d '' nested; do
      codesign --force --options runtime --timestamp \
        --entitlements "$HELPER_ENTITLEMENTS" -s "$DEVELOPER_ID_APP" "$nested"
    done
  codesign --force --options runtime --timestamp \
    --entitlements "$ENTITLEMENTS" -s "$DEVELOPER_ID_APP" "$APP"
  codesign --verify --deep --strict --verbose=2 "$APP"
else
  echo "[build_mac] no DEVELOPER_ID_APP set — AD-HOC signing for local dev."
  echo "             (TCC grants will reset on each rebuild; fine for launch testing,"
  echo "              NOT for distribution. Set DEVELOPER_ID_APP for a real build.)"
  # Always pass --entitlements: without allow-jit / disable-library-validation
  # QtWebEngine + several PyInstaller dylibs fail to load (dyld kill). Fail loud
  # rather than silently producing a bundle that crashes on first launch.
  codesign --force --deep -s - --entitlements "$ENTITLEMENTS" "$APP"
  codesign --verify --strict --verbose=2 "$APP"
fi

echo "[build_mac] built: $APP"

# ---- 6. optional installer + notarization ----------------------------------
if [[ "$DO_PKG" == "1" ]]; then
  NOTARG=""
  [[ "$DO_NOTARIZE" == "1" ]] && NOTARG="--notarize"
  bash "$ROOT/installers/macos/build_pkg.sh" $NOTARG
fi
if [[ "$DO_DMG" == "1" ]]; then
  bash "$ROOT/installers/macos/build_dmg.sh"
fi

echo "[build_mac] done."
