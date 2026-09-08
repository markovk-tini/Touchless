"""macOS system-audio mix helpers (platform-agnostic)."""

import numpy as np

from hgr.platform_compat.mac_system_audio import mix_mac_pcm


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
