"""r20: the always-on clip recorder should be off by default on hardware
that cannot carry it, and on by default everywhere else.

The recorder screen-grabs the whole desktop 20 times a second and hands
ffmpeg a full BGRA frame each time. On a 6.1 megapixel desktop driven by
a 2 GB GPU that measured out at roughly two CPU cores, competing with the
gesture pipeline, for a clip the user never asked for. On the reference
rig (5.2 megapixels, 12 GB) it is cheap and must stay on.
"""

import inspect

import pytest

from hgr.diagnostics import test_my_pc as T


REFERENCE_RIG = (3584 * 1440, 12282)   # measured on the dev machine
FIELD_RIG = (4240 * 1440, 2048)        # measured on the older machine


def test_the_reference_rig_keeps_its_clip_cache():
    """Large desktop, big GPU. This must not change."""
    assert T.should_disable_clip_cache(*REFERENCE_RIG) is False


def test_the_field_rig_is_seeded_off():
    assert T.should_disable_clip_cache(*FIELD_RIG) is True


def test_unknown_vram_fails_open():
    """A failed probe must never disable a feature."""
    assert T.should_disable_clip_cache(4240 * 1440, 0) is False
    assert T.should_disable_clip_cache(4240 * 1440, -1) is False


def test_a_weak_gpu_on_a_small_desktop_is_fine():
    assert T.should_disable_clip_cache(1920 * 1080, 2048) is False


def test_a_big_desktop_on_a_mid_gpu_is_fine():
    assert T.should_disable_clip_cache(4240 * 1440, 6144) is False


def test_both_halves_are_required():
    px_hi, px_lo = 4240 * 1440, 1280 * 720
    assert T.should_disable_clip_cache(px_hi, 2048) is True
    assert T.should_disable_clip_cache(px_lo, 2048) is False
    assert T.should_disable_clip_cache(px_hi, 8192) is False


@pytest.mark.parametrize("bad", [(None, 2048), ("wide", 2048), (4240 * 1440, "2 GB")])
def test_nonsense_inputs_fail_open(bad):
    assert T.should_disable_clip_cache(*bad) is False


def test_the_thresholds_are_named_not_inline():
    assert T.CLIP_CACHE_HEAVY_PIXELS == 3840 * 1200
    assert T.CLIP_CACHE_WEAK_VRAM_MB == 3072


# ------------------------------------------------------- the VRAM probe

def test_vram_probe_spawns_nothing():
    """The whole point is to adapt without paying probe_system()'s two
    PowerShell calls, which cost seconds and get scanned by antivirus."""
    src = inspect.getsource(T.discrete_gpu_vram_mb)
    for banned in ("subprocess", "powershell", "wmic", "Popen", "check_output"):
        assert banned not in src, banned
    assert "winreg" in src


def test_vram_probe_returns_a_non_negative_int():
    v = T.discrete_gpu_vram_mb()
    assert isinstance(v, int) and v >= 0


def test_vram_probe_beats_the_wmi_ceiling():
    """Win32_VideoController.AdapterRAM is a 32-bit field and reports any
    card of 4 GB or more as exactly 4095 MB. If this machine has such a
    card, the registry path must see past that."""
    v = T.discrete_gpu_vram_mb()
    if v == 0:
        pytest.skip("no adapter VRAM readable on this machine")
    assert v != 4095, "still reading the clamped WMI value"


# --------------------------------------------------------- the seeding

def test_the_seeder_runs_at_most_once_and_fails_open():
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from hgr.app.ui.main_window import MainWindow

    src = inspect.getsource(MainWindow._seed_clip_cache_default_once)
    assert "clip_cache_default_seeded" in src, "must latch so it decides once"
    assert "disable = False" in src, "must fail open on any probe error"
    for banned in ("subprocess", "powershell", "probe_system"):
        assert banned not in src, banned


def test_the_seeder_is_called_before_the_clip_cache_starts():
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from hgr.app.ui.main_window import MainWindow

    whole = inspect.getsource(MainWindow)
    i = whole.find("_seed_clip_cache_default_once()")
    j = whole.find('if bool(getattr(self.config, "clip_cache_enabled", True)):')
    assert i != -1 and j != -1
    assert i < j, "the default must be resolved before the flag is read"

# ------------------------------------------- integrated vs discrete

def test_integrated_adapters_are_recognised():
    for name in ("Intel(R) UHD Graphics 770", "Intel(R) HD Graphics 530",
                 "Intel(R) Iris(R) Xe Graphics", "AMD Radeon(TM) Graphics",
                 "Microsoft Basic Display Adapter"):
        assert T._looks_integrated(name) is True, name


def test_add_in_cards_are_not_treated_as_integrated():
    for name in ("NVIDIA GeForce RTX 4070", "NVIDIA GeForce GTX 960",
                 "AMD Radeon RX 6800 XT", "NVIDIA Quadro P2000"):
        assert T._looks_integrated(name) is False, name


def test_an_unnamed_adapter_is_not_assumed_integrated():
    assert T._looks_integrated("") is False
    assert T._looks_integrated(None) is False


def test_a_discrete_card_wins_over_a_bigger_integrated_reading(monkeypatch):
    """The reason this matters: a plain maximum would report a shared-memory
    iGPU figure and hide a weak discrete card, which would silently skip the
    clip-cache seeding on exactly the machines that need it."""
    adapters = [("Intel(R) HD Graphics 530", 4096 * 1024 * 1024),
                ("NVIDIA GeForce GTX 960", 2048 * 1024 * 1024)]
    _install_fake_registry(monkeypatch, adapters)
    assert T.discrete_gpu_vram_mb() == 2048
    assert T.should_disable_clip_cache(4240 * 1440, 2048) is True


def test_an_integrated_only_machine_still_reports_something(monkeypatch):
    adapters = [("Intel(R) UHD Graphics 630", 2048 * 1024 * 1024)]
    _install_fake_registry(monkeypatch, adapters)
    assert T.discrete_gpu_vram_mb() == 2048


def test_adapters_with_no_memory_value_are_skipped(monkeypatch):
    adapters = [("Intel(R) UHD Graphics 770", None),
                ("NVIDIA GeForce RTX 4070", 12878610432)]
    _install_fake_registry(monkeypatch, adapters)
    assert T.discrete_gpu_vram_mb() == 12282


def _install_fake_registry(monkeypatch, adapters):
    """Stand in for the display-class registry key."""
    import contextlib

    class FakeWinreg:
        HKEY_LOCAL_MACHINE = object()

        @staticmethod
        @contextlib.contextmanager
        def OpenKey(root, sub, *a, **kw):
            yield ("root", sub) if root is FakeWinreg.HKEY_LOCAL_MACHINE else sub

        @staticmethod
        def EnumKey(key, i):
            if i >= len(adapters):
                raise OSError
            return str(i).zfill(4)

        @staticmethod
        def QueryValueEx(key, name):
            idx = int(key[1]) if isinstance(key, tuple) else int(key)
            desc, mem = adapters[idx]
            if name == "DriverDesc":
                return (desc, 1)
            if mem is None:
                raise OSError
            return (mem, 11)

    import sys as _sys
    monkeypatch.setitem(_sys.modules, "winreg", FakeWinreg)
    monkeypatch.setattr(T.sys, "platform", "win32")
