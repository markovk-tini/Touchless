"""macOS system-audio mix helpers (platform-agnostic)."""

import numpy as np

from hgr.platform_compat.mac_system_audio import (
    MAC_CLIP_STAMP_LEAD_S,
    MAC_MIX_MIC_GAIN,
    assemble_pcm_ring,
    boost_quiet_mac_pcm,
    mac_clip_mux_plan,
    mac_clip_video_timescale,
    mac_mux_atempo_factor,
    mix_mac_pcm,
    polish_mac_pcm,
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


def test_boost_quiet_mac_pcm_lifts_half_volume_recording():
    # Screen recordings were ~half live level (peak ~0.35). Default
    # already_loud=0.28 skipped that. Recording mux uses 0.50 / 0.85.
    half = np.full(4800, 0.35, dtype=np.float32)
    boosted = boost_quiet_mac_pcm(
        half, target_peak=0.85, max_gain=3.5, already_loud=0.50
    )
    assert boosted is not None
    peak = float(np.max(np.abs(boosted)))
    assert peak > 0.70
    assert peak <= 0.86


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


def test_mac_clip_stamp_lead_is_zero():
    # Constant delay made 30 s clips late; rate is atempo's job.
    assert abs(float(MAC_CLIP_STAMP_LEAD_S) - 0.0) < 1e-9


def test_mac_clip_mux_plan_locks_av_across_preset_lengths():
    # 1 s of packed samples on every clip preset: start stays at t=0
    # and atempo fills the picture so the end is not ~1 s early.
    for wall in (30.0, 60.0, 120.0, 300.0):
        packed = wall - 1.0
        scale, tempo = mac_clip_mux_plan(packed, wall, wall)
        assert scale == 1.0
        assert abs(tempo - (packed / wall)) < 1e-6
        assert abs(packed / tempo - wall) < 1e-6


def test_mac_clip_mux_plan_stretches_video_then_atempo_audio_to_wall():
    # OpenCV under-tags 50 s for a 60 s window; PCM is 1 s short.
    scale, tempo = mac_clip_mux_plan(59.0, 50.0, 60.0)
    assert abs(scale - 1.2) < 1e-6
    assert abs(tempo - (59.0 / 60.0)) < 1e-6


def test_looks_like_mp4_rejects_ffmpeg_log(tmp_path):
    from hgr.platform_compat.mac_system_audio import looks_like_mp4

    log = tmp_path / "Touchless_Recording_1.mp4.ffmpeg.log"
    log.write_text(
        "ffmpeg version\n  Duration: N/A, start: 31133.913833, bitrate: N/A\n",
        encoding="utf-8",
    )
    assert looks_like_mp4(log) is False

    tiny = tmp_path / "tiny.mp4"
    tiny.write_bytes(b"not a video")
    assert looks_like_mp4(tiny) is False

    mp4 = tmp_path / "Touchless_Recording_1.mp4"
    # ISO-BMFF: 4-byte box size + 'ftyp'
    mp4.write_bytes(b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 64)
    assert looks_like_mp4(mp4) is True


def test_mix_mac_pcm_ducks_room_mic_when_both_present():
    fs = 48000
    mic = np.ones(fs, dtype=np.float32) * 0.5
    sys_a = np.ones(fs, dtype=np.float32) * 0.2
    unducked = mix_mac_pcm(mic, sys_a)
    ducked = mix_mac_pcm(mic, sys_a, mic_gain=MAC_MIX_MIC_GAIN)
    assert unducked is not None and ducked is not None
    assert abs(float(np.mean(unducked)) - 0.7) < 0.02
    # 0.5 * mic_gain + 0.2
    assert abs(float(np.mean(ducked)) - (0.5 * MAC_MIX_MIC_GAIN + 0.2)) < 0.02
    # Mic-only still full level (no SCK to comb against).
    mic_only = mix_mac_pcm(mic, None, mic_gain=MAC_MIX_MIC_GAIN)
    assert abs(float(np.mean(mic_only)) - 0.5) < 0.01


def test_polish_mac_pcm_fades_edges_keeps_middle():
    fs = 48000
    buf = np.full(fs, 0.6, dtype=np.float32)
    out = polish_mac_pcm(buf, fs=fs, fade_s=0.010)
    assert out is not None
    n = int(0.010 * fs)
    assert float(out[0]) < 0.02
    assert float(out[-1]) < 0.02
    assert abs(float(np.mean(out[n * 2 : -n * 2])) - 0.6) < 0.02


def test_assemble_pcm_ring_dropout_pad_keeps_duration():
    fs = 48000
    a = np.full(fs, 0.5, dtype=np.float32)
    b = np.full(fs, 0.5, dtype=np.float32)
    # 400 ms gap is a real dropout (> 250 ms).
    chunks = [(1.0, a), (2.4, b)]
    out = assemble_pcm_ring(chunks, fs, 0.0, 2.4)
    assert out is not None
    assert len(out) == int(round(2.4 * fs))
    # Gap is near-silent; we fade 5 ms into it, so don't require a
    # perfectly empty middle sample.
    mid = out[fs + int(0.1 * fs) : fs + int(0.3 * fs)]
    assert float(np.max(np.abs(mid))) < 0.05


def test_assemble_pcm_ring_unpadded_is_shorter_than_window_when_packed():
    fs = 48000
    # 0.9 s of samples stamped as covering 1.0 s (typical xrun pack).
    a = np.full(int(0.9 * fs), 0.4, dtype=np.float32)
    chunks = [(1.0, a)]
    padded = assemble_pcm_ring(chunks, fs, 0.0, 1.0)
    unpadded = assemble_pcm_ring(chunks, fs, 0.0, 1.0, pad_to_window=False)
    assert padded is not None and unpadded is not None
    assert len(padded) == fs
    assert len(unpadded) == int(0.9 * fs)


def test_mac_mux_atempo_slows_packed_audio_onto_video():
    # 59 s of samples vs 60 s picture → slow down ~1.7% (the ~1 s early case).
    tempo = mac_mux_atempo_factor(59.0, 60.0)
    assert abs(tempo - (59.0 / 60.0)) < 1e-6
    assert tempo < 1.0
    assert mac_mux_atempo_factor(60.0, 60.0) == 1.0
    assert mac_mux_atempo_factor(60.0, 1.0) == 1.0  # implausible, skip
    # 1 s pack on 5 min is 0.33% — still correct it (120 ms floor).
    assert abs(mac_mux_atempo_factor(299.0, 300.0) - (299.0 / 300.0)) < 1e-6
    assert mac_mux_atempo_factor(60.0, 60.05) == 1.0  # 50 ms, ignore


def test_polish_mac_pcm_removes_one_sample_click():
    fs = 48000
    buf = np.full(4000, 0.1, dtype=np.float32)
    buf[2000] = 0.95
    out = polish_mac_pcm(buf, fs=fs, fade_s=0.0)
    assert out is not None
    assert float(out[2000]) < 0.3
