"""Touchless application package.

Single source of truth for the app version. Inno Setup's MyAppVersion
in installers/windows/hgr_app.iss MUST be kept in sync — the auto-
updater compares the running app's __version__ against the GitHub
release tag, and the installer writes the same string into the
Add/Remove Programs entry.
"""

__version__ = "1.1.8"

# v1.1.7 build-round marker. Bump on each shipped-source polish round so
# the build_marker_label in Settings → About (and any future diagnostics
# HUD) pull from one source of truth instead of a scattered literal.
BUILD_ROUND = 51

# Author: Konstantin Markov
