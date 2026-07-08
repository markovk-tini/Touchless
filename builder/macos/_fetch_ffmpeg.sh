#!/usr/bin/env bash
# Fetch a STATIC arm64 macOS ffmpeg + ffprobe into builder/macos/vendor/ so the
# .pkg bundles a self-contained ffmpeg and recording / clip audio works on ANY
# user's Mac with ZERO setup (the shipping contract in CLAUDE.md rule #6).
#
# Why not just use the build machine's ffmpeg: a Homebrew ffmpeg is DYNAMICALLY
# linked against /opt/homebrew dylibs the end user does not have, so shipping it
# produces a binary that crashes on first launch. A static build is
# self-contained.
#
# Source: Martin Riedl's static macOS builds (stable "latest" redirects). If
# that host is down or you prefer another static source (e.g. osxexperts.net),
# override with FFMPEG_URL / FFPROBE_URL. Run once before build_mac.sh; the
# spec (hgr_app_mac.spec) then bundles whatever is in vendor/.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENDOR="$HERE/vendor"
mkdir -p "$VENDOR"

FFMPEG_URL="${FFMPEG_URL:-https://ffmpeg.martin-riedl.de/redirect/latest/macos/arm64/release/ffmpeg.zip}"
FFPROBE_URL="${FFPROBE_URL:-https://ffmpeg.martin-riedl.de/redirect/latest/macos/arm64/release/ffprobe.zip}"

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "[fetch-ffmpeg] macOS only (uname is '$(uname -s)')."
  exit 1
fi

fetch_tool() {
  local name="$1" url="$2"
  echo "[fetch-ffmpeg] downloading $name from $url"
  local tmp
  tmp="$(mktemp -d)"
  curl -fsSL "$url" -o "$tmp/$name.zip"
  unzip -o -q "$tmp/$name.zip" -d "$tmp"
  if [[ ! -f "$tmp/$name" ]]; then
    # Some archives nest the binary a level down.
    local found
    found="$(find "$tmp" -type f -name "$name" | head -n1)"
    [[ -n "$found" ]] && mv "$found" "$tmp/$name"
  fi
  if [[ ! -f "$tmp/$name" ]]; then
    echo "[fetch-ffmpeg] ERROR: $name binary not found in archive"
    rm -rf "$tmp"
    exit 1
  fi
  chmod +x "$tmp/$name"
  mv -f "$tmp/$name" "$VENDOR/$name"
  rm -rf "$tmp"
  # Sanity checks: arm64 Mach-O, self-contained (no Homebrew/local dylibs).
  if ! file "$VENDOR/$name" | grep -q "arm64"; then
    echo "[fetch-ffmpeg] WARN: $name is not arm64 — wrong architecture for Apple Silicon."
  fi
  if otool -L "$VENDOR/$name" | grep -qE "/opt/homebrew|/usr/local/(lib|opt|Cellar)"; then
    echo "[fetch-ffmpeg] WARN: $name links Homebrew/local dylibs — NOT static; it will"
    echo "               fail on end-user Macs. Use a fully static build."
  else
    echo "[fetch-ffmpeg] OK: $name is self-contained ($(du -h "$VENDOR/$name" | cut -f1))"
  fi
}

fetch_tool ffmpeg "$FFMPEG_URL"
fetch_tool ffprobe "$FFPROBE_URL"
echo "[fetch-ffmpeg] done -> $VENDOR (these are gitignored; re-run to update)"
