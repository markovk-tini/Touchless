"""Per-mic-class voice tuning profiles.

Classifies a microphone by its sounddevice `name` (plus optional rate +
host-API metadata) into one of a small set of MicClass values, and
returns a MicProfile dataclass of tuning parameters tuned for that
class. The VoiceCommandListener uses the profile to set per-class
defaults for suggested gain, VAD trigger floors, AGC target, and the
end-of-utterance window — instead of one-size-fits-all hard-coded
constants that work poorly for low-gain webcam mics or hot USB
condensers.

Pure stdlib (re, dataclasses, enum). No I/O. Callers pass the
sounddevice metadata they already pulled — keeps the module easy to
unit-test without a physical mic.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class MicClass(str, Enum):
    WEBCAM = "webcam"
    HEADSET = "headset"
    USB_CONDENSER = "usb_condenser"
    PHONE = "phone"
    BLUETOOTH = "bluetooth"
    LAPTOP_BUILTIN = "laptop_builtin"
    GENERIC = "generic"


# Order matters: more specific patterns first. Each (class, regex). The
# regex is searched against the device's Windows-friendly name.
# Sources: real device-name strings observed in sd.query_devices() on
# Windows WASAPI for the most common consumer mics shipped with this
# app's user base. Add a row + a test fixture when a new mic class
# becomes worth tuning for.
_PATTERNS: tuple[tuple[MicClass, "re.Pattern[str]"], ...] = (
    # Phone sentinel: Touchless's own phone connector exposes a
    # specific device name.
    (MicClass.PHONE, re.compile(
        r"(?:^|\b)(touchless\s*phone|phone\s*\(qr\)|phone\s*camera\s*mic)\b",
        re.I,
    )),
    # USB condenser / streaming mics (hot signal, clip-prone).
    (MicClass.USB_CONDENSER, re.compile(
        r"\b(blue\s*yeti|yeti(?:\s*nano|\s*x)?|rode\s*(?:nt[-\s]?usb|podmic)|"
        r"elgato\s*wave|shure\s*mv\d+|samson\s*g[\-\s]?track|"
        r"hyperx\s*quadcast|fifine|maono|at2020(?:usb)?)\b", re.I,
    )),
    # Bluetooth HFP headsets MUST be checked BEFORE plain "headset"
    # pattern — many BT devices ship as e.g. "Headset (WH-1000XM4
    # Hands-Free AG)" and the generic headset regex would falsely
    # catch them first. HFP narrowband needs the bluetooth profile.
    (MicClass.BLUETOOTH, re.compile(
        r"\b(airpods|wh[-\s]?1000xm|wf[-\s]?1000xm|galaxy\s*buds|"
        r"pixel\s*buds|jabra\s*elite|hands[-\s]?free(?:\s*ag)?|"
        r"\(hfp\)|bluetooth\s*(?:hands[-\s]?free|hf)|bose\s*(?:qc|qcii)|"
        r"beats(?:\s*studio)?)\b", re.I,
    )),
    # USB / wired gaming headsets.
    (MicClass.HEADSET, re.compile(
        r"\b(hyperx\s*cloud|steelseries\s*arctis|astro\s*a\d+|"
        r"corsair\s*hs\d+|logitech\s*g\s*pro\s*x|razer\s*(?:kraken|"
        r"barracuda|blackshark)|sennheiser\s*(?:gsp|pc\d+)|"
        r"epos\s*(?:gsp|h\d+)|jabra\s*evolve|plantronics|poly\s*blackwire|"
        r"headset(?:\s*microphone)?)\b", re.I,
    )),
    # Webcam mics (omnidirectional, low gain, far-field).
    (MicClass.WEBCAM, re.compile(
        r"\b(razer\s*kiyo(?:\s*pro)?|logitech\s*(?:c\d{3,4}|brio|streamcam)|"
        r"insta360\s*link|obsbot|hd\s*(?:pro\s*)?webcam|webcam\s*c\d+|"
        r"facetime\s*hd|web\s*cam(?:era)?)\b", re.I,
    )),
    # Built-in laptop arrays — Realtek / Conexant / Intel SST.
    (MicClass.LAPTOP_BUILTIN, re.compile(
        r"\b(realtek(?:\(r\))?\s*audio|conexant|intel\s*smart\s*sound|"
        r"sst|microphone\s*array|internal\s*microphone|"
        r"built[-\s]?in\s*microphone)\b", re.I,
    )),
)


@dataclass(frozen=True)
class MicProfile:
    """Per-class tuning parameters used by the voice listener.

    Fields are deliberately conservative — the goal is "works first
    time on a fresh install with this mic class". Power users can
    override the slider, which sets the global `mic_input_gain_auto`
    to False (see app_config.py) so suggested_gain is no longer
    auto-applied.
    """
    mic_class: MicClass
    suggested_gain: float        # default multiplier in voice listener
    trigger_floor: float         # absolute minimum trigger_threshold (overrides 0.005)
    silence_floor: float         # absolute minimum silence_threshold (overrides 0.003)
    agc_target_rms: float        # whisper-preprocessing target RMS (overrides 0.10)
    agc_max_gain: float          # whisper-preprocessing AGC cap (overrides 6.0)
    end_silence_seconds: float   # per-class end-of-utterance window
    preroll_seconds: float       # per-class preroll duration
    expects_clipping: bool       # if True, apply pre-normalize attenuation
    narrowband_warning: bool     # surface a one-line stderr warning (HFP)
    display_label: str           # user-facing description for UI badge


# Per-class suggested gains were initially set aggressively (e.g.
# WEBCAM=3.0) on the theory that webcam mics are quiet — but modern
# webcams like the Razer Kiyo Pro have hot built-in preamps where
# 3.0× catastrophically clips (peak > 4.0). Defaults are now
# conservative: a slight bump for genuinely quiet hardware (laptop
# array), a slight cut for known-hot hardware (USB condensers), and
# 1.0× neutral for everything where the Windows volume is the right
# place to tune. The runtime clipping auto-attenuator (see
# `_maybe_attenuate_for_clipping` in voice_command_listener) cuts gain
# in half on the next attempt when peak > 0.95 — that's the safety
# net for users whose Windows level is already cranked.
_DEFAULTS: dict[MicClass, MicProfile] = {
    MicClass.WEBCAM: MicProfile(
        MicClass.WEBCAM, suggested_gain=1.0, trigger_floor=0.004,
        silence_floor=0.0025, agc_target_rms=0.10, agc_max_gain=6.0,
        end_silence_seconds=0.8, preroll_seconds=2.0,
        expects_clipping=False, narrowband_warning=False,
        display_label="Webcam mic"),
    MicClass.HEADSET: MicProfile(
        MicClass.HEADSET, suggested_gain=1.0, trigger_floor=0.006,
        silence_floor=0.004, agc_target_rms=0.10, agc_max_gain=3.0,
        end_silence_seconds=0.8, preroll_seconds=1.5,
        expects_clipping=False, narrowband_warning=False,
        display_label="Headset mic (close-field)"),
    MicClass.USB_CONDENSER: MicProfile(
        MicClass.USB_CONDENSER, suggested_gain=0.7, trigger_floor=0.008,
        silence_floor=0.005, agc_target_rms=0.10, agc_max_gain=2.0,
        end_silence_seconds=0.8, preroll_seconds=1.5,
        expects_clipping=True, narrowband_warning=False,
        display_label="USB studio mic (hot signal)"),
    MicClass.PHONE: MicProfile(
        MicClass.PHONE, suggested_gain=1.0, trigger_floor=0.004,
        silence_floor=0.0025, agc_target_rms=0.10, agc_max_gain=4.0,
        end_silence_seconds=1.0, preroll_seconds=2.0,
        expects_clipping=False, narrowband_warning=False,
        display_label="Phone microphone"),
    MicClass.BLUETOOTH: MicProfile(
        MicClass.BLUETOOTH, suggested_gain=1.0, trigger_floor=0.005,
        silence_floor=0.003, agc_target_rms=0.10, agc_max_gain=4.0,
        end_silence_seconds=0.9, preroll_seconds=1.8,
        expects_clipping=False, narrowband_warning=True,
        display_label="Bluetooth headset (narrowband)"),
    MicClass.LAPTOP_BUILTIN: MicProfile(
        MicClass.LAPTOP_BUILTIN, suggested_gain=1.3, trigger_floor=0.004,
        silence_floor=0.0025, agc_target_rms=0.10, agc_max_gain=5.0,
        end_silence_seconds=0.9, preroll_seconds=2.0,
        expects_clipping=False, narrowband_warning=False,
        display_label="Built-in laptop microphone"),
    MicClass.GENERIC: MicProfile(
        MicClass.GENERIC, suggested_gain=1.0, trigger_floor=0.005,
        silence_floor=0.003, agc_target_rms=0.10, agc_max_gain=6.0,
        end_silence_seconds=0.8, preroll_seconds=2.0,
        expects_clipping=False, narrowband_warning=False,
        display_label="Microphone"),
}


def classify_mic(
    device_name: Optional[str],
    *,
    sample_rate: Optional[int] = None,
    host_api_name: Optional[str] = None,
    max_input_channels: Optional[int] = None,
    is_external_phone: bool = False,
) -> MicProfile:
    """Classify a microphone into a `MicClass` and return its tuning
    profile.

    Caller supplies what it already pulled from `sd.query_devices()`
    or the phone connector. Pure function — no I/O, no sounddevice
    import. Always returns a profile (never None); falls back to
    `MicClass.GENERIC` when no pattern + metadata matches.

    Args:
        device_name: Windows-friendly device name like
            ``"Microphone (Razer Kiyo Pro)"``.
        sample_rate: Device's default sample rate. Rates < 16000 are
            promoted to BLUETOOTH (HFP narrowband) when no name pattern
            matched.
        host_api_name: Currently informational only.
        max_input_channels: Currently informational only.
        is_external_phone: True when the listener is driving the
            external phone audio source. Forces PHONE regardless of
            name patterns (the phone connector already pre-boosts the
            signal worklet-side; PHONE's suggested_gain accounts for
            that combined target).
    """
    if is_external_phone:
        return _DEFAULTS[MicClass.PHONE]

    name = (device_name or "").strip()
    cls = MicClass.GENERIC

    # Name-pattern pass (most signal).
    for candidate, pattern in _PATTERNS:
        if pattern.search(name):
            cls = candidate
            break

    # Metadata override — but ONLY if the name patterns didn't
    # already classify. Some pro audio interfaces report odd default
    # rates and shouldn't be misclassified as Bluetooth.
    if cls is MicClass.GENERIC and sample_rate is not None and sample_rate > 0 and sample_rate < 16000:
        cls = MicClass.BLUETOOTH

    return _DEFAULTS[cls]


def profile_for_class(mic_class: MicClass) -> MicProfile:
    """Direct accessor for tests and overrides."""
    return _DEFAULTS[mic_class]
