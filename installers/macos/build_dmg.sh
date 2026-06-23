#!/usr/bin/env bash
# Touchless — macOS drag-to-Applications .dmg builder (website download channel).
# NEW, SEPARATE from the Windows installer. Run after build_mac.sh produces the .app.
#
#     ./installers/macos/build_dmg.sh
#
# Prefers `create-dmg` (brew install create-dmg) for the styled drag layout;
# falls back to plain `hdiutil` if it isn't installed.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

APP="$ROOT/dist/Touchless.app"
[[ -d "$APP" ]] || { echo "ERROR: $APP not found. Run builder/macos/build_mac.sh first."; exit 1; }
VERSION="${TOUCHLESS_VERSION:-1.0.0}"
DMG="$ROOT/dist/Touchless-$VERSION.dmg"
rm -f "$DMG"

# create-dmg copies the CONTENTS of its source_folder into the image, so we must
# stage Touchless.app inside a folder and pass the FOLDER — otherwise the .app's
# Contents/ tree ends up loose at the DMG root and the drag layout breaks.
STAGING="$(mktemp -d)"
cp -R "$APP" "$STAGING/"

if command -v create-dmg >/dev/null 2>&1; then
  echo "[build_dmg] using create-dmg ..."
  create-dmg \
    --volname "Touchless" \
    --app-drop-link 450 160 \
    --icon "Touchless.app" 150 160 \
    --window-size 600 360 \
    "$DMG" "$STAGING"
else
  echo "[build_dmg] create-dmg not found — using hdiutil (plain layout). brew install create-dmg for a styled one."
  ln -s /Applications "$STAGING/Applications"
  hdiutil create -volname "Touchless" -srcfolder "$STAGING" -ov -format UDZO "$DMG"
fi
rm -rf "$STAGING"

# Sign the .dmg with a timestamp if a Developer ID is available (helps Gatekeeper).
if [[ -n "${DEVELOPER_ID_APP:-}" ]]; then
  codesign --force --timestamp --sign "$DEVELOPER_ID_APP" "$DMG"
fi

# Notarize + staple the .dmg so a website download isn't quarantined by
# Gatekeeper. Requires the App Store Connect API creds (same as build_pkg.sh).
# The .app inside MUST already be signed with --options runtime (build_mac.sh does).
if [[ -n "${AC_API_KEY_ID:-}" && -n "${AC_API_ISSUER_ID:-}" && -n "${AC_API_KEY_P8:-}" ]]; then
  echo "[build_dmg] notarytool submit (waits for Apple) ..."
  xcrun notarytool submit "$DMG" \
    --key "$AC_API_KEY_P8" --key-id "$AC_API_KEY_ID" --issuer "$AC_API_ISSUER_ID" --wait
  xcrun stapler staple "$DMG"
  xcrun stapler validate "$DMG"
else
  echo "[build_dmg] NOTE: AC_API_* not set — DMG is NOT notarized. A downloaded,"
  echo "            un-notarized DMG will be quarantined by Gatekeeper. Set the"
  echo "            AC_API_* creds for a distributable DMG."
fi

echo "[build_dmg] built: $DMG"
