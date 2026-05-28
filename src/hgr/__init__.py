"""Touchless application package.

Single source of truth for the app version. Inno Setup's MyAppVersion
in installers/windows/hgr_app.iss MUST be kept in sync — the auto-
updater compares the running app's __version__ against the GitHub
release tag, and the installer writes the same string into the
Add/Remove Programs entry.
"""

__version__ = "1.1.2"

# Author: Konstantin Markov
