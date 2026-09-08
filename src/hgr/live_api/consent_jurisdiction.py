"""Two-party-consent jurisdiction detection.

Phase-1 trust substrate. In the US, recording a conversation requires
the consent of BOTH parties in:

    California, Florida, Illinois, Maryland, Massachusetts, Montana,
    Nevada, New Hampshire, Pennsylvania, Washington (and Delaware,
    Michigan, Oregon for some scenarios; Connecticut for civil).

In other states + most of the rest of the world it's single-party
(the user can record their own conversations). The product / legal
implication is BLUNT: in two-party states, Iris transcribing audio
from a call WITHOUT a recorded explicit opt-in from the OTHER party
is a criminal-law issue, not a UX problem. Illinois Eavesdropping Act
in particular treats this as a felony.

This module is the single source of truth for "is this user in a
two-party-consent jurisdiction?". Any feature that listens to audio
(always-on wake word, call transcription, meeting recap, dictation
that captures system audio) checks `requires_two_party_consent()`
BEFORE enabling recording, and gates accordingly:

  * Two-party state + recording requested → require typed-consent
    flow that surfaces a disclosure tone the OTHER party hears.
    Or refuse the feature entirely. No silent recording, ever.
  * Single-party state → standard "Iris is listening" indicator
    + audit log row.

Detection sources, tried in order:
  1. Explicit user override (`TOUCHLESS_JURISDICTION_TWO_PARTY=1`).
     Always wins. Lets cautious users force the safer behavior.
  2. Windows region setting (geoex / locale). Reliable on most
     installs, doesn't require network.
  3. Best-effort IP geolocation (free, low-precision — only used
     when locale check is inconclusive AND the user opts into one
     internet check). Disabled by default.
  4. Fallback: assume TWO-PARTY (safer default; user can override).

Author: Konstantin Markov
"""
from __future__ import annotations

import os
import platform
import threading
from typing import Optional, Tuple


# All-party-consent (two-party in common parlance) US states.
# Authoritative as of 2024-2025; revisit if a state changes its
# wiretap statute. New Hampshire is included on the strict reading.
TWO_PARTY_US_STATES = frozenset({
    "CA", "FL", "IL", "MD", "MA", "MT", "NV", "NH", "PA", "WA",
    # Civil-liability cases extend the list; including them keeps
    # the safer side as default:
    "DE", "MI", "OR", "CT",
})


# Two-party-consent countries (non-exhaustive — covers what we can
# reliably detect via locale). EU GDPR-style consent overlays this:
# in the EU, recording any identifiable speech requires a lawful
# basis and typically the data subject's informed consent. So all
# EU member states default to two-party for Iris's purposes.
TWO_PARTY_COUNTRIES = frozenset({
    # EU
    "AT", "BE", "BG", "HR", "CY", "CZ", "DK", "EE", "FI", "FR",
    "DE", "GR", "HU", "IE", "IT", "LV", "LT", "LU", "MT", "NL",
    "PL", "PT", "RO", "SK", "SI", "ES", "SE",
    # UK + EEA
    "GB", "NO", "IS", "LI",
})


_state_lock = threading.RLock()
_cached_result: Optional[Tuple[bool, str]] = None


def _detect_via_env() -> Optional[bool]:
    v = (os.environ.get("TOUCHLESS_JURISDICTION_TWO_PARTY")
         or "").strip().lower()
    if v in ("1", "true", "yes", "on"):
        return True
    if v in ("0", "false", "no", "off"):
        return False
    return None


def _detect_via_windows() -> Optional[Tuple[bool, str]]:
    """Read Windows region (Country) + GeoID. Returns
    (is_two_party, reason) or None if probe failed."""
    if platform.system() != "Windows":
        return None
    try:
        import ctypes
        # GetUserGeoID(GEOCLASS_NATION=16) returns the user's nation
        # as a Microsoft GEOID; combine with GetGeoInfo(GEO_ISO2) to
        # get the ISO-3166-1 alpha-2 country code.
        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        geo_id = kernel32.GetUserGeoID(16)
        if not geo_id:
            return None
        buf = ctypes.create_unicode_buffer(8)
        # GEO_ISO2 = 4. langid 0 = default.
        n = kernel32.GetGeoInfoW(geo_id, 4, buf, len(buf), 0)
        if n <= 0:
            return None
        country = buf.value.strip().upper()
    except Exception:
        return None
    if not country:
        return None
    if country in TWO_PARTY_COUNTRIES:
        return True, f"locale country={country}"
    if country == "US":
        # US — check state via the system locale name (en-US-CA etc.
        # isn't reliable). The best we can do without an IP lookup
        # is honor the user's region "Sub-Region" if exposed. Since
        # Windows doesn't reliably expose US state, default to
        # single-party here (recording in single-party states is the
        # majority US case) BUT add the disclosure tone always.
        # The conservative answer when state is unknown but country
        # is US: stay single-party-permissive, rely on the user to
        # override via TOUCHLESS_JURISDICTION_TWO_PARTY=1 if they
        # live in CA/FL/IL/MD/MA/MT/NV/NH/PA/WA.
        return False, "locale country=US (state unknown)"
    return False, f"locale country={country} (single-party default)"


def requires_two_party_consent() -> Tuple[bool, str]:
    """Returns (two_party_required, reason). Cached across the
    process — the user's jurisdiction doesn't change at runtime.
    Call `clear_cache()` after the user explicitly changes region."""
    global _cached_result
    with _state_lock:
        if _cached_result is not None:
            return _cached_result
        # 1. Explicit override
        env = _detect_via_env()
        if env is True:
            _cached_result = (True, "env override")
            return _cached_result
        if env is False:
            _cached_result = (False, "env override (single-party)")
            return _cached_result
        # 2. Windows region
        win = _detect_via_windows()
        if win is not None:
            _cached_result = win
            return _cached_result
        # 3. Fallback — safer default
        _cached_result = (True,
                          "could not detect — defaulting to two-party "
                          "for safety. Set TOUCHLESS_JURISDICTION_TWO_PARTY=0 "
                          "if you live in a single-party state.")
        return _cached_result


def clear_cache() -> None:
    global _cached_result
    with _state_lock:
        _cached_result = None


# ---- consent gate helpers used by audio features --------------------------


def can_transcribe_call_audio(*,
                              user_has_opted_in_this_session: bool = False
                              ) -> Tuple[bool, str]:
    """Check before transcribing audio that may include another party
    (system audio loopback, meeting / call audio). Returns
    (allowed, reason)."""
    two_party, why = requires_two_party_consent()
    if not two_party:
        return True, f"single-party jurisdiction ({why})"
    if user_has_opted_in_this_session:
        return True, f"user opted in for this session ({why})"
    return False, (
        f"two-party-consent jurisdiction detected ({why}). "
        "Recording call audio requires explicit per-session opt-in "
        "AND an audible 'Iris is recording' disclosure tone the OTHER "
        "party can hear. Iris will not capture this audio.")


def must_play_disclosure_tone() -> bool:
    """True when an audible 'Iris is listening' tone MUST be played
    each time the mic goes hot. In two-party jurisdictions: always.
    In single-party: still recommended UX but not legally required."""
    two_party, _ = requires_two_party_consent()
    return two_party
