#!/usr/bin/env bash
# Touchless — macOS dev run / test launcher.
#
# NEW, SEPARATE from the Windows flow (run_test.py / builder/windows/*). This
# script never touches the Windows build or its .venv. It creates a dedicated
# .venv-mac, installs requirements_mac.txt, and runs Touchless from source.
#
# Usage:
#   ./run_mac.sh                 # set up the venv (if needed) and launch the app
#   ./run_mac.sh --smoke         # run the headless import/capability smoke test only
#   ./run_mac.sh --reinstall     # force pip to reinstall deps, then launch
#   ./run_mac.sh --no-install    # skip the pip step (fast relaunch)
#   PYTHON=python3.11 ./run_mac.sh   # pick a specific interpreter
#
# Requires: a recent python3 (3.10–3.12 recommended) on the PATH.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

VENV="$ROOT/.venv-mac"
PYBIN="${PYTHON:-python3}"
DO_INSTALL=1
SMOKE_ONLY=0
REINSTALL=0

for arg in "$@"; do
  case "$arg" in
    --smoke)      SMOKE_ONLY=1 ;;
    --no-install) DO_INSTALL=0 ;;
    --reinstall)  REINSTALL=1 ;;
    -h|--help)    grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "run_mac.sh: unknown arg '$arg' (try --help)"; exit 2 ;;
  esac
done

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "run_mac.sh is for macOS only (uname is '$(uname -s)'). On Windows use builder/windows/."
  exit 1
fi

if ! command -v "$PYBIN" >/dev/null 2>&1; then
  echo "ERROR: '$PYBIN' not found. Install Python 3 (e.g. 'brew install python@3.12') or set PYTHON=..."
  exit 1
fi

# ---- venv -------------------------------------------------------------------
if [[ ! -d "$VENV" ]]; then
  echo "[run_mac] creating venv at $VENV"
  "$PYBIN" -m venv "$VENV"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"
VPY="$VENV/bin/python"

# ---- deps -------------------------------------------------------------------
STAMP="$VENV/.deps-installed"
if [[ "$REINSTALL" == "1" || ( "$DO_INSTALL" == "1" && ! -f "$STAMP" ) ]]; then
  echo "[run_mac] installing requirements_mac.txt (this can take a few minutes the first time)"
  "$VPY" -m pip install --upgrade pip wheel
  if ! "$VPY" -m pip install -r "$ROOT/requirements_mac.txt"; then
    echo ""
    echo "[run_mac] pip install failed. The most common cause is the mediapipe pin"
    echo "          not having an arm64 macOS wheel. Try bumping mediapipe in"
    echo "          requirements_mac.txt to the newest 0.10.x, or report the error."
    exit 1
  fi
  touch "$STAMP"
else
  echo "[run_mac] using existing deps (pass --reinstall to refresh)"
fi

# ---- static ffmpeg (recording audio) ---------------------------------------
# Same self-contained ffmpeg the .pkg bundles, vendored under builder/macos/
# vendor/ so a SOURCE run gets off-thread recording + mic audio with NO
# `brew install` (matches the zero-setup shipped app). Non-fatal.
if [[ ! -x "$ROOT/builder/macos/vendor/ffmpeg" ]]; then
  echo "[run_mac] fetching static ffmpeg (recording audio; one-time) ..."
  bash "$ROOT/builder/macos/_fetch_ffmpeg.sh" || \
    echo "[run_mac] ffmpeg fetch failed (recording will be disabled until it's present)."
fi

# ---- run --------------------------------------------------------------------
if [[ "$SMOKE_ONLY" == "1" ]]; then
  exec "$VPY" "$ROOT/scripts/mac_smoke_test.py"
fi

echo "[run_mac] launching Touchless from source ..."
exec "$VPY" "$ROOT/run_app.py"
