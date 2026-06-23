#!/usr/bin/env bash
# Touchless — build whisper.cpp with Metal (Apple Silicon GPU) acceleration.
#
# NEW, SEPARATE from builder/windows/_build_whisper_cuda.bat /
# _build_whisper_vulkan.bat (those stay untouched). Replaces the Windows
# CUDA/Vulkan path with Metal. Produces arm64 Mach-O binaries under
# whisper.cpp/build_metal/bin/ (no .exe, no bin/Release) — the layout the macOS
# PyInstaller spec (_collect_metal_runtime) and the voice runtime finder expect.
#
# This is part of the VOICE phase, not the base build. The base build runs fine
# on faster-whisper (CPU/Accelerate) without it. Invoked by build_mac.sh --metal.
#
# Requires: Xcode command line tools (clang), cmake. Optional: Homebrew SDL2 for
# the streaming example (`brew install sdl2`).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WHISPER="$ROOT/whisper.cpp"
[[ -d "$WHISPER" ]] || { echo "[whisper-metal] $WHISPER not found — clone whisper.cpp there first."; exit 1; }

command -v cmake >/dev/null 2>&1 || { echo "[whisper-metal] cmake not found (brew install cmake)."; exit 1; }

BUILD="$WHISPER/build_metal"
echo "[whisper-metal] configuring (Metal ON) ..."
cmake -S "$WHISPER" -B "$BUILD" \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_METAL=ON \
  -DGGML_METAL_EMBED_LIBRARY=ON \
  -DWHISPER_BUILD_EXAMPLES=ON \
  -DCMAKE_OSX_ARCHITECTURES=arm64

echo "[whisper-metal] building ..."
cmake --build "$BUILD" --config Release -j

echo "[whisper-metal] binaries in $BUILD/bin :"
ls -1 "$BUILD/bin" 2>/dev/null || true
echo "[whisper-metal] done."
