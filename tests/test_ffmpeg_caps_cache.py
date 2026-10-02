"""r20: the ffmpeg capability probe result is cached across launches.

The probe spawns eight ffmpeg.exe processes on the GUI thread at every
launch. That is seconds of startup on an older machine and, on a box
running consumer antivirus, eight fresh scans that the user sees as
popups. The answer only changes when the GPU, its driver, or the
bundled binary changes, so it is cached against a fingerprint of
exactly those.

These tests drive the cache helpers directly rather than constructing a
MainWindow, which keeps them fast and headless.
"""

import os

import pytest

from hgr.app.ui.main_window import MainWindow


class Cfg:
    def __init__(self):
        self.ffmpeg_caps_cache_key = ""
        self.ffmpeg_caps_cache = {}


class Stub:
    """Only the attributes the cache helpers touch."""

    _FFMPEG_CAPS_PROBE_VERSION = MainWindow._FFMPEG_CAPS_PROBE_VERSION
    _display_adapter_fingerprint = MainWindow._display_adapter_fingerprint
    _ffmpeg_caps_cache_key = MainWindow._ffmpeg_caps_cache_key
    _load_cached_ffmpeg_capabilities = MainWindow._load_cached_ffmpeg_capabilities

    def __init__(self, ffmpeg_path):
        self._ffmpeg_path = ffmpeg_path
        self.config = Cfg()


@pytest.fixture
def stub(tmp_path):
    exe = tmp_path / "ffmpeg.exe"
    exe.write_bytes(b"x" * 64)
    return Stub(str(exe))


GOOD = {
    "available": True,
    "encoders": ["h264_nvenc", "libx264"],
    "filters": ["ddagrab"],
    "devices": ["gdigrab"],
    "preferred_encoder": "h264_nvenc",
    "nvenc_modern_presets": True,
}


# ------------------------------------------------------------ the key

def test_key_is_stable_for_an_unchanged_binary(stub):
    assert stub._ffmpeg_caps_cache_key() == stub._ffmpeg_caps_cache_key()


def test_key_is_non_empty_and_names_its_inputs(stub):
    key = stub._ffmpeg_caps_cache_key()
    assert key
    assert "ffmpeg.exe" in key and "gpu=" in key
    assert key.startswith(f"v{MainWindow._FFMPEG_CAPS_PROBE_VERSION}|")


def test_key_changes_when_the_binary_changes(stub):
    before = stub._ffmpeg_caps_cache_key()
    with open(stub._ffmpeg_path, "wb") as fh:
        fh.write(b"y" * 4096)
    os.utime(stub._ffmpeg_path, (1_700_000_000, 1_700_000_000))
    assert stub._ffmpeg_caps_cache_key() != before


def test_no_key_without_an_ffmpeg_path(stub):
    stub._ffmpeg_path = ""
    assert stub._ffmpeg_caps_cache_key() == ""


def test_probe_version_is_part_of_the_key(stub):
    before = stub._ffmpeg_caps_cache_key()
    stub._FFMPEG_CAPS_PROBE_VERSION = MainWindow._FFMPEG_CAPS_PROBE_VERSION + 1
    assert stub._ffmpeg_caps_cache_key() != before


# ---------------------------------------------------------- hit / miss

def test_a_matching_key_returns_the_cached_answer(stub):
    key = stub._ffmpeg_caps_cache_key()
    stub.config.ffmpeg_caps_cache_key = key
    stub.config.ffmpeg_caps_cache = dict(GOOD)
    caps = stub._load_cached_ffmpeg_capabilities(key)
    assert caps["preferred_encoder"] == "h264_nvenc"
    assert caps["encoders"] == {"h264_nvenc", "libx264"}
    assert isinstance(caps["filters"], set)


def test_a_different_key_is_a_miss(stub):
    stub.config.ffmpeg_caps_cache_key = "some-older-key"
    stub.config.ffmpeg_caps_cache = dict(GOOD)
    assert stub._load_cached_ffmpeg_capabilities(stub._ffmpeg_caps_cache_key()) is None


def test_an_empty_key_is_always_a_miss(stub):
    stub.config.ffmpeg_caps_cache_key = ""
    stub.config.ffmpeg_caps_cache = dict(GOOD)
    assert stub._load_cached_ffmpeg_capabilities("") is None


def test_a_cached_negative_result_is_never_trusted(stub):
    """Caching 'ffmpeg is missing' would hide a repaired install."""
    key = stub._ffmpeg_caps_cache_key()
    stub.config.ffmpeg_caps_cache_key = key
    stub.config.ffmpeg_caps_cache = {"available": False}
    assert stub._load_cached_ffmpeg_capabilities(key) is None


def test_a_cached_result_with_no_encoders_is_rejected(stub):
    key = stub._ffmpeg_caps_cache_key()
    stub.config.ffmpeg_caps_cache_key = key
    stub.config.ffmpeg_caps_cache = {"available": True, "encoders": []}
    assert stub._load_cached_ffmpeg_capabilities(key) is None


def test_a_corrupt_cache_is_a_miss_not_a_crash(stub):
    key = stub._ffmpeg_caps_cache_key()
    stub.config.ffmpeg_caps_cache_key = key
    stub.config.ffmpeg_caps_cache = "not a dict"
    assert stub._load_cached_ffmpeg_capabilities(key) is None


# ------------------------------------------------------------ wiring

def test_the_probe_body_still_exists_separately():
    """_detect_ffmpeg_capabilities is now a thin cache wrapper; the real
    probing must still live in its own method so the cache can be
    bypassed."""
    assert callable(getattr(MainWindow, "_probe_ffmpeg_capabilities", None))


def test_a_runtime_demotion_invalidates_the_cache():
    import inspect

    whole = inspect.getsource(MainWindow)
    i = whole.find("_clip_export_encoder_demoted = True")
    assert i != -1
    assert "invalidate_ffmpeg_capabilities_cache" in whole[i:i + 800], (
        "a runtime encoder demotion must drop the cached probe result"
    )


def test_the_modern_preset_flag_survives_the_round_trip(stub):
    """It picks -preset p4 over legacy medium at encode time. Losing it
    silently downgrades every clip on a capable GPU."""
    key = stub._ffmpeg_caps_cache_key()
    stub.config.ffmpeg_caps_cache_key = key
    stub.config.ffmpeg_caps_cache = dict(GOOD)
    caps = stub._load_cached_ffmpeg_capabilities(key)
    assert caps["nvenc_modern_presets"] is True


def test_a_payload_missing_any_probe_field_is_rejected(stub):
    key = stub._ffmpeg_caps_cache_key()
    for missing in ("encoders", "filters", "devices", "preferred_encoder",
                    "nvenc_modern_presets"):
        payload = dict(GOOD)
        payload.pop(missing)
        stub.config.ffmpeg_caps_cache_key = key
        stub.config.ffmpeg_caps_cache = payload
        assert stub._load_cached_ffmpeg_capabilities(key) is None, missing


def test_encoders_rehydrate_as_a_mutable_set(stub):
    """The runtime watchdog calls .discard() on it."""
    key = stub._ffmpeg_caps_cache_key()
    stub.config.ffmpeg_caps_cache_key = key
    stub.config.ffmpeg_caps_cache = dict(GOOD)
    caps = stub._load_cached_ffmpeg_capabilities(key)
    assert type(caps["encoders"]) is set
    caps["encoders"].discard("h264_nvenc")


def test_the_key_includes_the_build_round(stub):
    from hgr import BUILD_ROUND

    assert f"|br={BUILD_ROUND}|" in stub._ffmpeg_caps_cache_key()


def test_a_degraded_probe_is_not_cached(stub):
    """A sub-probe timeout is not an answer. Caching one would turn a
    one-off antivirus scanning delay into a permanent downgrade."""
    stub._store_ffmpeg_capabilities_cache = MainWindow._store_ffmpeg_capabilities_cache.__get__(stub)
    key = stub._ffmpeg_caps_cache_key()
    caps = dict(GOOD)
    caps["encoders"] = {"libx264"}
    caps["probe_degraded"] = True
    stub._store_ffmpeg_capabilities_cache(key, caps)
    assert stub.config.ffmpeg_caps_cache_key == ""
    assert stub.config.ffmpeg_caps_cache == {}


def test_probe_degraded_is_never_persisted(stub, monkeypatch):
    import hgr.app.ui.main_window as MW

    monkeypatch.setattr(MW, "save_config", lambda cfg: None)
    stub._store_ffmpeg_capabilities_cache = MainWindow._store_ffmpeg_capabilities_cache.__get__(stub)
    key = stub._ffmpeg_caps_cache_key()
    caps = dict(GOOD)
    caps["encoders"] = {"h264_nvenc", "libx264"}
    caps["probe_degraded"] = False
    stub._store_ffmpeg_capabilities_cache(key, caps)
    assert "probe_degraded" not in stub.config.ffmpeg_caps_cache
    assert stub.config.ffmpeg_caps_cache["nvenc_modern_presets"] is True


def test_the_probe_declares_the_degraded_flag_up_front():
    import inspect

    src = inspect.getsource(MainWindow._probe_ffmpeg_capabilities)
    head = src[: src.index("if not self._ffmpeg_path")]
    assert '"probe_degraded": False' in head, (
        "the flag must be in the initial dict so handler order cannot matter"
    )
    assert src.count('capabilities["probe_degraded"] = True') >= 6, (
        "every sub-probe timeout handler must mark the run degraded"
    )

def test_the_clip_cache_watchdog_demote_also_invalidates():
    import inspect

    whole = inspect.getsource(MainWindow)
    i = whole.find('self._ffmpeg_capabilities["preferred_encoder"] = next_pref')
    assert i != -1
    assert "invalidate_ffmpeg_capabilities_cache" in whole[i:i + 800], (
        "the clip-cache watchdog demote must drop the cached probe result"
    )

# -------------------------------- a failed probe is not an answer

def test_a_result_with_no_encoders_is_never_stored(stub, monkeypatch):
    """_run_external_probe returns an empty string for ANY failure,
    timeout included. ffmpeg never legitimately lists zero encoders, so
    an empty list means the probe did not run. Caching it would leave
    hardware encoding off and silently re-probe on every future launch."""
    import hgr.app.ui.main_window as MW

    monkeypatch.setattr(MW, "save_config", lambda cfg: None)
    stub._store_ffmpeg_capabilities_cache = MainWindow._store_ffmpeg_capabilities_cache.__get__(stub)
    key = stub._ffmpeg_caps_cache_key()
    caps = dict(GOOD)
    caps["encoders"] = set()
    stub._store_ffmpeg_capabilities_cache(key, caps)
    assert stub.config.ffmpeg_caps_cache_key == ""
    assert stub.config.ffmpeg_caps_cache == {}


def test_an_empty_enumeration_marks_the_run_degraded():
    import inspect

    src = inspect.getsource(MainWindow._probe_ffmpeg_capabilities)
    for which in ("encoders_text", "filters_text", "devices_text"):
        i = src.index(which + " = self._run_external_probe")
        window = src[i:i + 1000]
        assert "probe_degraded" in window, (
            which + " must mark the run degraded when it comes back empty"
        )
