"""macOS system-audio mix helpers (platform-agnostic)."""

import numpy as np

from hgr.platform_compat.mac_system_audio import (
    boost_quiet_mac_pcm,
    mac_clip_video_timescale,
    mix_mac_pcm,
)


def test_mix_end_aligns_longer_system_buffer():
    fs = 48000
    mic = np.ones(fs, dtype=np.float32) * 0.5
    # 2 s of system audio: first second is a marker, second matches mic.
    sys_a = np.concatenate(
        [
            np.full(fs, 0.9, dtype=np.float32),
            np.full(fs, 0.1, dtype=np.float32),
        ]
    )
    mixed = mix_mac_pcm(mic, sys_a)
    assert mixed is not None
    assert len(mixed) == 2 * fs
    # Start-align would put 0.9 into the first second of the mix.
    # End-align keeps the marker in the first second and mic+0.1 in the last.
    assert float(np.mean(mixed[:fs])) > 0.8
    assert 0.4 < float(np.mean(mixed[fs:])) < 0.8


def test_mac_clip_video_timescale_stretches_short_probe():
    # 50 s tagged file vs 60 s wall → ~10 s soundtrack lead if we
    # end-trim audio to the probe instead of stretching video.
    scale = mac_clip_video_timescale(50.0, 60.0)
    assert abs(scale - 1.2) < 1e-6
    assert mac_clip_video_timescale(60.0, 60.0) == 1.0
    assert mac_clip_video_timescale(59.5, 60.0) == 1.0
    assert mac_clip_video_timescale(0.0, 60.0) == 1.0


def test_boost_quiet_mac_pcm_raises_sck_level():
    quiet = np.full(4800, 0.04, dtype=np.float32)
    boosted = boost_quiet_mac_pcm(quiet)
    assert boosted is not None
    assert float(np.max(np.abs(boosted))) > 0.45
    loud = np.full(4800, 0.5, dtype=np.float32)
    same = boost_quiet_mac_pcm(loud)
    assert float(np.max(np.abs(same))) == 0.5
