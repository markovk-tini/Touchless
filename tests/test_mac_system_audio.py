"""macOS system-audio mix helpers (platform-agnostic)."""

import numpy as np

from hgr.platform_compat.mac_system_audio import (
    MAC_CLIP_STAMP_LEAD_S,
    assemble_pcm_ring,
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
    assert float(np.max(np.abs(boosted))) > 0.12
    loud = np.full(4800, 0.5, dtype=np.float32)
    same = boost_quiet_mac_pcm(loud)
    assert float(np.max(np.abs(same))) == 0.5


def test_assemble_pcm_ring_concat_keeps_stamp_overlap_samples():
    fs = 48000
    a = np.full(fs, 0.2, dtype=np.float32)
    # Stamps say this block overlaps the first by 50 ms; samples are
    # still sequential. Crossfading them mixed 0.2 with 0.9 (static)
    # and shortened the ring. Concat keeps both blocks.
    b = np.full(int(0.1 * fs), 0.9, dtype=np.float32)
    chunks = [(1.0, a), (1.05, b)]
    out = assemble_pcm_ring(chunks, fs, 0.0, 1.05)
    assert out is not None
    assert len(out) == int(round(1.05 * fs))
    assert float(np.mean(out[int(0.1 * fs) : int(0.8 * fs)])) < 0.3
    # First 50 ms of b land at t=1.0s in the concat.
    assert float(np.mean(out[fs : fs + int(0.04 * fs)])) > 0.7


def test_assemble_pcm_ring_first_start_window():
    fs = 48000
    # 3 s ending at t=10: two seconds of 0.1 then a 1 s 0.9 marker.
    # Window [7, 9] is the first two seconds (first-start = 7).
    marker = np.full(fs, 0.9, dtype=np.float32)
    rest = np.full(2 * fs, 0.1, dtype=np.float32)
    chunks = [(10.0, np.concatenate([rest, marker]))]
    out = assemble_pcm_ring(chunks, fs, 7.0, 9.0)
    assert out is not None
    assert len(out) == 2 * fs
    assert float(np.mean(out)) < 0.2


def test_assemble_pcm_ring_negative_stamp_lead_includes_older_audio():
    fs = 48000
    early = np.full(fs, 0.9, dtype=np.float32)
    later = np.full(2 * fs, 0.1, dtype=np.float32)
    chunks = [(10.0, np.concatenate([early, later]))]
    # first_start=7. Window [8, 10] at lead=0 is 2 s of 0.1.
    # lead=-1 pulls in the 0.9 marker.
    out = assemble_pcm_ring(chunks, fs, 8.0, 10.0, stamp_lead_s=-1.0)
    assert out is not None
    assert len(out) == 2 * fs
    assert float(np.mean(out[:fs])) > 0.7
    assert float(np.mean(out[fs:])) < 0.2


def test_assemble_pcm_ring_positive_stamp_lead_skips_older_audio():
    fs = 48000
    early = np.full(fs, 0.9, dtype=np.float32)
    later = np.full(2 * fs, 0.1, dtype=np.float32)
    chunks = [(10.0, np.concatenate([early, later]))]
    out = assemble_pcm_ring(chunks, fs, 7.0, 9.0, stamp_lead_s=1.0)
    assert out is not None
    assert len(out) == 2 * fs
    assert float(np.mean(out)) < 0.2


def test_assemble_pcm_ring_ignores_sub_quarter_second_gap():
    fs = 48000
    a = np.full(fs, 0.5, dtype=np.float32)
    b = np.full(fs, 0.5, dtype=np.float32)
    # 80 ms stamp gap used to insert a click of zeros.
    chunks = [(1.0, a), (2.08, b)]
    out = assemble_pcm_ring(chunks, fs, 0.0, 2.0)
    assert out is not None
    assert float(np.min(np.abs(out))) > 0.4


def test_mac_clip_stamp_lead_delays_early_soundtrack():
    assert abs(float(MAC_CLIP_STAMP_LEAD_S) - (-1.5)) < 1e-9
