"""Touchless application package.

Single source of truth for the app version. Inno Setup's MyAppVersion
in installers/windows/hgr_app.iss MUST be kept in sync — the auto-
updater compares the running app's __version__ against the GitHub
release tag, and the installer writes the same string into the
Add/Remove Programs entry.
"""

__version__ = "1.1.9.2"

# v1.1.7 build-round marker. Bump on each shipped-source polish round so
# the build_marker_label in Settings → About (and any future diagnostics
# HUD) pull from one source of truth instead of a scattered literal.
#
# r24: this is now PURELY an identity marker and is safe to bump on every
# build. It used to be mixed into the camera-capability cache key, which
# meant bumping it silently discarded everything the r21 probe had learned
# about the user's webcam -- so r22 and r23 both shipped without bumping it,
# and a field log became impossible to attribute to a build. Cache
# invalidation now lives in CAMERA_CAPS_PROBE_VERSION below. BUMP THIS
# EVERY BUILD.
BUILD_ROUND = 64

# Cache key for the r21 camera-capability probe
# (`ffmpeg -list_options`). Bump ONLY when the probe's command, its parser
# or the shape it stores changes -- never for an unrelated code change.
# What a camera reports it can do does not depend on our build number, and
# re-learning costs the user an ffmpeg spawn and an antivirus prompt.
CAMERA_CAPS_PROBE_VERSION = 1

# Author: Konstantin Markov
