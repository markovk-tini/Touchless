#!/usr/bin/env bash
# Touchless — build llama.cpp with Metal (Apple Silicon GPU) acceleration.
#
# NEW, SEPARATE from builder/windows/build_llama_cuda.bat (untouched). Replaces
# the Windows CUDA path with Metal. Produces arm64 Mach-O binaries (llama-server,
# llama-cli) under llama.cpp/build_metal/bin/ — the layout the macOS PyInstaller
# spec and the grammar corrector / Iris local backend expect.
#
# Part of the VOICE phase, not the base build. Invoked by build_mac.sh --metal.
# Requires: Xcode command line tools (clang), cmake.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LLAMA="$ROOT/llama.cpp"
[[ -d "$LLAMA" ]] || { echo "[llama-metal] $LLAMA not found — clone llama.cpp there first."; exit 1; }

command -v cmake >/dev/null 2>&1 || { echo "[llama-metal] cmake not found (brew install cmake)."; exit 1; }

BUILD="$LLAMA/build_metal"
echo "[llama-metal] configuring (Metal ON) ..."
cmake -S "$LLAMA" -B "$BUILD" \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_METAL=ON \
  -DGGML_METAL_EMBED_LIBRARY=ON \
  -DLLAMA_BUILD_SERVER=ON \
  -DCMAKE_OSX_ARCHITECTURES=arm64

echo "[llama-metal] building ..."
cmake --build "$BUILD" --config Release -j

echo "[llama-metal] binaries in $BUILD/bin :"
ls -1 "$BUILD/bin" 2>/dev/null || true
echo "[llama-metal] done."
