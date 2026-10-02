"""OPEN_ISSUES 1.6: D3D11 capture frames need an explicit download.

`ddagrab` and `gfxcapture` emit frames in GPU memory. NVENC consumes
those directly; `libx264` cannot, and QSV cannot on some rigs -- the
encoder then produces ZERO segments, the watchdog demotes, and the clip
cache dies. Live since r5 (2026-09-15) and never caught, because it only
reproduces on SINGLE-MONITOR machines (a multi-monitor desktop gives
`monitor_index=None` and falls through to gdigrab, already a CPU path)
and both test rigs are multi-monitor NVIDIA.

The two properties that matter, and why each is a separate case below:
the download MUST appear for every non-NVENC encoder, or the feature is
dead on those machines; and it must NOT appear for NVENC, or every
NVIDIA user pays a per-frame GPU->CPU copy for nothing.

Driven unbound against a minimal stub, matching test_ffmpeg_caps_cache.
"""
from __future__ import annotations

import pytest
from PySide6.QtCore import QRect

from hgr.app.ui.main_window import MainWindow


DOWNLOAD = ",hwdownload,format=bgra"


class Stub:
    """Only what the capture-args helpers touch."""

    def __init__(self, encoder="libx264", filters=("ddagrab",), monitor_index=0):
        self._ffmpeg_capabilities = {
            "preferred_encoder": encoder,
            "filters": set(filters),
        }
        self._monitor_index = monitor_index

    def _matching_monitor_index(self, _rect):
        return self._monitor_index

    def _ffmpeg_gpu_capture_suffix(self):
        # Delegate to the REAL implementation rather than returning a
        # canned string, so the graph cases below exercise the actual
        # encoder logic and not a test double of it.
        return MainWindow._ffmpeg_gpu_capture_suffix(self)


def _suffix(encoder):
    return MainWindow._ffmpeg_gpu_capture_suffix(Stub(encoder=encoder))


def _graph(encoder, filters, monitor_index=0):
    args = MainWindow._ffmpeg_capture_input_args(
        Stub(encoder=encoder, filters=filters, monitor_index=monitor_index),
        QRect(0, 0, 1920, 1080),
        fps=10.0,
        prefer_low_overhead=True,
    )
    return args


# ---- the suffix itself -------------------------------------------------

def test_nvenc_gets_no_download():
    """NVENC takes D3D11 frames directly. Adding the copy would cost
    every NVIDIA user frame rate for nothing."""
    assert _suffix("h264_nvenc") == ""


@pytest.mark.parametrize("encoder", ["libx264", "h264_qsv", "h264_amf"])
def test_non_nvenc_encoders_get_the_download(encoder):
    """libx264 cannot take GPU frames at all; QSV cannot on some rigs
    (OPEN_ISSUES 1.6 names it explicitly), and AMF is untested here. The
    asymmetry is deliberate: a needless copy costs frame rate, a missing
    one costs the whole feature."""
    assert _suffix(encoder) == DOWNLOAD


def test_unknown_encoder_defaults_to_downloading():
    """Fail safe, not fast. An encoder we have never seen is assumed to
    need system memory."""
    assert _suffix("h264_something_new") == DOWNLOAD


def test_missing_capabilities_default_to_downloading():
    stub = Stub()
    stub._ffmpeg_capabilities = {}
    assert MainWindow._ffmpeg_gpu_capture_suffix(stub) == DOWNLOAD


def test_broken_capabilities_object_defaults_to_downloading():
    """The helper is called while building a subprocess command line; it
    must not raise there."""
    class Exploding:
        def get(self, *_a, **_k):
            raise RuntimeError("caps unavailable")

    stub = Stub()
    stub._ffmpeg_capabilities = Exploding()
    assert MainWindow._ffmpeg_gpu_capture_suffix(stub) == DOWNLOAD


# ---- the suffix reaching the real lavfi graph --------------------------

def test_ddagrab_graph_carries_the_download_for_libx264():
    args = _graph("libx264", ("ddagrab",))
    assert args[:3] == ["-f", "lavfi", "-i"]
    assert args[3].startswith("ddagrab=output_idx=0")
    assert args[3].endswith(DOWNLOAD), args[3]


def test_ddagrab_graph_omits_the_download_for_nvenc():
    args = _graph("h264_nvenc", ("ddagrab",))
    assert args[3].startswith("ddagrab=output_idx=0")
    assert "hwdownload" not in args[3], args[3]


def test_gfxcapture_graph_carries_it_too():
    """gfxcapture is tried BEFORE ddagrab and is also a D3D11 filter, so
    fixing only ddagrab would have left the bug live wherever gfxcapture
    is available."""
    args = _graph("libx264", ("gfxcapture", "ddagrab"))
    assert args[3].startswith("gfxcapture=monitor_idx=0")
    assert args[3].endswith(DOWNLOAD), args[3]


def test_gfxcapture_graph_omits_it_for_nvenc():
    args = _graph("h264_nvenc", ("gfxcapture", "ddagrab"))
    assert "hwdownload" not in args[3], args[3]


# ---- the CPU path must be untouched ------------------------------------

def test_gdigrab_fallback_is_unaffected():
    """Multi-monitor desktops resolve `monitor_index=None` and use
    gdigrab, which already delivers CPU frames. This is the path both
    test rigs take, so it is also the regression guard for the
    BUILD_ROUND 62 save point.
    """
    args = _graph("libx264", ("ddagrab",), monitor_index=None)
    assert "gdigrab" in args
    assert not any("hwdownload" in str(a) for a in args), args


def test_gdigrab_used_when_no_gpu_filter_is_available():
    args = _graph("libx264", (), monitor_index=0)
    assert "gdigrab" in args
    assert not any("hwdownload" in str(a) for a in args), args


# Author: Konstantin Markov
