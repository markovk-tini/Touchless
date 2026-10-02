"""r20: the ffmpeg fallback must bound its own wall-clock cost and must
report HOW it failed, without changing the successful-open path.

The reference rig opens on the first candidate in about a second. These
tests pin that this path never consults the budget and never reports a
failure, so the slow-rig fix cannot regress it.
"""

import pytest

from hgr.app.camera import ffmpeg_capture as F
from hgr.app.camera.ffmpeg_memo import KIND_BUSY, KIND_HARD, KIND_SILENT


class FakeCap:
    """Stands in for FfmpegMjpegCapture. `plan` drives each attempt."""

    instances = []

    def __init__(self, device_name, width=0, height=0, fps=0):
        self.fps = fps
        self.released = False
        outcome = FakeCap.plan.pop(0) if FakeCap.plan else ("hard", 0.0)
        self.kind, self.cost = outcome
        FakeCap.clock[0] += self.cost
        FakeCap.instances.append(self)

    def isOpened(self):
        return self.kind == "ok"

    def _last_failure_was_silent_hang(self):
        return self.kind == "silent"

    def release(self):
        self.released = True

    def read(self):
        return False, None


@pytest.fixture(autouse=True)
def _fake(monkeypatch):
    FakeCap.instances = []
    FakeCap.plan = []
    FakeCap.clock = [1000.0]
    monkeypatch.setattr(F, "FfmpegMjpegCapture", FakeCap)
    monkeypatch.setattr(F.time, "monotonic", lambda: FakeCap.clock[0])
    monkeypatch.setattr(F.time, "sleep", lambda s: FakeCap.clock.__setitem__(0, FakeCap.clock[0] + s))


def _call(**kw):
    out = {}
    cap = F.open_ffmpeg_cap_with_fps_fallback("Cam", failure_out=out, **kw)
    return cap, out


# ------------------------------------------------- the fast-rig path

def test_first_candidate_success_is_untouched():
    FakeCap.plan = [("ok", 1.0)]
    cap, out = _call()
    assert cap is not None and cap.fps == 60
    assert len(FakeCap.instances) == 1      # exactly one spawn
    assert out == {}                        # no failure reported


def test_success_is_unaffected_by_a_tiny_budget():
    """The budget is only consulted before spending MORE time, so a
    camera that opens immediately cannot be starved by it."""
    FakeCap.plan = [("ok", 1.0)]
    cap, out = _call(total_budget_seconds=0.0)
    assert cap is not None and out == {}


# ------------------------------------------------------ the budget

def test_budget_stops_the_second_candidate():
    FakeCap.plan = [("hard", 5.0), ("ok", 1.0)]
    cap, out = _call(total_budget_seconds=3.0)
    assert cap is None
    assert len(FakeCap.instances) == 1      # never reached 30 fps
    assert out["kind"] == KIND_HARD


def test_without_the_budget_both_candidates_run():
    FakeCap.plan = [("hard", 1.0), ("ok", 1.0)]
    cap, out = _call(total_budget_seconds=60.0)
    assert cap is not None and cap.fps == 30
    assert len(FakeCap.instances) == 2


def test_budget_stops_the_silent_hang_retry():
    FakeCap.plan = [("silent", 9.0), ("ok", 1.0)]
    cap, out = _call(total_budget_seconds=4.0)
    assert cap is None
    assert len(FakeCap.instances) == 1
    assert out["kind"] == KIND_BUSY


# ------------------------------------------------- failure kinds

def test_a_format_rejection_reports_hard():
    FakeCap.plan = [("hard", 0.1), ("hard", 0.1)]
    cap, out = _call()
    assert cap is None and out["kind"] == KIND_HARD


def test_handle_races_alone_never_report_hard():
    FakeCap.plan = [("silent", 0.1), ("silent", 0.1)]
    cap, out = _call()
    assert cap is None and out["kind"] in (KIND_BUSY, KIND_SILENT)
    assert out["kind"] != KIND_HARD


def test_one_hard_then_a_hang_still_reports_hard():
    FakeCap.plan = [("hard", 0.1), ("silent", 0.1), ("silent", 0.1)]
    cap, out = _call()
    assert cap is None and out["kind"] == KIND_HARD


def test_silent_hang_retry_can_still_succeed():
    FakeCap.plan = [("silent", 0.1), ("ok", 0.1)]
    cap, out = _call()
    assert cap is not None and out == {}


def test_failure_out_is_optional():
    FakeCap.plan = [("hard", 0.1), ("hard", 0.1)]
    assert F.open_ffmpeg_cap_with_fps_fallback("Cam") is None
