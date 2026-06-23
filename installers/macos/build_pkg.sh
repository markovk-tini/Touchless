#!/usr/bin/env bash
# Touchless — macOS .pkg installer builder.
#
# NEW, SEPARATE from installers/windows/hgr_app.iss (Inno Setup). Wraps
# dist/Touchless.app into a signed, optionally-notarized .pkg that installs to
# /Applications. Run after builder/macos/build_mac.sh has produced the .app.
#
#     ./installers/macos/build_pkg.sh [--notarize]
#
# Signing/notarization use the same env vars as build_mac.sh:
#     DEVELOPER_ID_INSTALLER, AC_API_KEY_ID, AC_API_ISSUER_ID, AC_API_KEY_P8
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

DO_NOTARIZE=0
[[ "${1:-}" == "--notarize" ]] && DO_NOTARIZE=1

APP="$ROOT/dist/Touchless.app"
[[ -d "$APP" ]] || { echo "ERROR: $APP not found. Run builder/macos/build_mac.sh first."; exit 1; }

VERSION="${TOUCHLESS_VERSION:-1.0.0}"
BUNDLE_ID="${TOUCHLESS_BUNDLE_ID:-com.touchless.app}"
# distribution.xml pins the component pkg-ref id to com.touchless.app.pkg. If the
# bundle id is overridden, productbuild can no longer resolve the component pkg.
# Fail fast rather than emit a broken installer.
if [[ "$BUNDLE_ID" != "com.touchless.app" ]]; then
  echo "ERROR: TOUCHLESS_BUNDLE_ID is '$BUNDLE_ID' but installers/macos/distribution.xml"
  echo "       is pinned to 'com.touchless.app'. Update distribution.xml's pkg-ref/choice"
  echo "       ids to match, or unset TOUCHLESS_BUNDLE_ID, before building the .pkg."
  exit 1
fi
OUT_DIR="$ROOT/dist"
COMPONENT_PKG="$OUT_DIR/Touchless-component.pkg"
PRODUCT_PKG="$OUT_DIR/Touchless-$VERSION.pkg"

# ---- 1. component pkg (app -> /Applications) --------------------------------
echo "[build_pkg] pkgbuild ..."
pkgbuild \
  --component "$APP" \
  --install-location "/Applications" \
  --identifier "$BUNDLE_ID.pkg" \
  --version "$VERSION" \
  "$COMPONENT_PKG"

# ---- 2. product archive (welcome/license/conclusion + arch requirement) ----
echo "[build_pkg] productbuild ..."
DISTRIBUTION="$ROOT/installers/macos/distribution.xml"
RESOURCES="$ROOT/installers/macos/resources"
# distribution.xml references welcome/license/conclusion.html, which productbuild
# resolves against --resources. The dir is required (it ships in the repo).
if [[ ! -d "$RESOURCES" ]]; then
  echo "ERROR: $RESOURCES not found, but distribution.xml references welcome/license/"
  echo "       conclusion.html. productbuild would fail. Restore installers/macos/resources/."
  exit 1
fi
PRODUCT_ARGS=(--distribution "$DISTRIBUTION" --package-path "$OUT_DIR" --resources "$RESOURCES")

if [[ -n "${DEVELOPER_ID_INSTALLER:-}" ]]; then
  UNSIGNED="$OUT_DIR/Touchless-$VERSION-unsigned.pkg"
  productbuild "${PRODUCT_ARGS[@]}" "$UNSIGNED"
  echo "[build_pkg] productsign with Developer ID Installer ..."
  productsign --sign "$DEVELOPER_ID_INSTALLER" "$UNSIGNED" "$PRODUCT_PKG"
  rm -f "$UNSIGNED"
else
  echo "[build_pkg] no DEVELOPER_ID_INSTALLER set — building UNSIGNED .pkg (dev only)."
  productbuild "${PRODUCT_ARGS[@]}" "$PRODUCT_PKG"
fi
rm -f "$COMPONENT_PKG"
echo "[build_pkg] built: $PRODUCT_PKG"

# ---- 3. notarize + staple --------------------------------------------------
if [[ "$DO_NOTARIZE" == "1" ]]; then
  if [[ -z "${AC_API_KEY_ID:-}" || -z "${AC_API_ISSUER_ID:-}" || -z "${AC_API_KEY_P8:-}" ]]; then
    echo "ERROR: --notarize needs AC_API_KEY_ID, AC_API_ISSUER_ID, AC_API_KEY_P8."
    exit 1
  fi
  echo "[build_pkg] notarytool submit (this waits for Apple) ..."
  xcrun notarytool submit "$PRODUCT_PKG" \
    --key "$AC_API_KEY_P8" --key-id "$AC_API_KEY_ID" --issuer "$AC_API_ISSUER_ID" \
    --wait
  echo "[build_pkg] stapling ..."
  xcrun stapler staple "$PRODUCT_PKG"
  xcrun stapler validate "$PRODUCT_PKG"
fi

echo "[build_pkg] done."
