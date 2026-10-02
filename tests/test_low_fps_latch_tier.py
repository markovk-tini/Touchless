"""r24: the auto-low-fps tier must come from which gate fired.

The field rig sat at ~10.3 fps and reported that Default, Lite, GPU and
Boost were all identical. They were, literally: the app had auto-engaged
low-fps mode and could never leave, and `if self._low_fps_active:` is the
FIRST branch of `_build_engine_for_fps_mode`, so all three modes built
the same 384-wide complexity-0 engine.

The latch was permanent because of a second-guess. `_engage_auto_low_fps`
re-derived the tier as `fps < _CRITICAL_FPS_THRESHOLD -
_CRITICAL_FPS_ENGAGE_MARGIN` (12 - 4 = 8). At 10.3 fps that is False, so
a machine that had just tripped the CRITICAL gate was filed as
"fullscreen" -- a tier needing 18 fps to exit, and one the hard-escape
block explicitly skips (`if tier == "critical"`). The guard existed to
stop transient dips latching the critical tier, but transients are
already handled upstream by _CRITICAL_FPS_ENTER_SECONDS, which requires
six continuous seconds under threshold.
"""

import pytest

import hgr.app.integration.noop_engine as NE

W = NE.GestureWorker


def _worker():
    w = W.__new__(W)
    w._fps = 0.0
    w._low_fps_auto_engaged = False
    w._low_fps_engaged_at = None
    w._low_fps_engaged_tier = None
    w._low_fps_below_since = None
    w._low_fps_above_since = None
    w._low_fps_hard_escape_since = None
    w._cap = None
    w._running = False
    w.config = type("C", (), {"low_fps_mode": False, "lite_mode": False,
                              "gpu_mode": False})()
    # neutralise the side effects: we are testing the tier decision only
    w._swap_engine_safely = lambda: None
    w._apply_low_fps_capture_tuning = lambda cap: None
    w._apply_perf_camera_path = lambda **kw: None
    return w


class TestTheTierFollowsTheGate:
    def test_the_critical_gate_yields_the_critical_tier(self):
        w = _worker()
        w._fps = 10.3          # the field rig's exact number
        w._engage_auto_low_fps(reason="critical")
        assert w._low_fps_engaged_tier == "critical"

    def test_the_fullscreen_gate_yields_the_fullscreen_tier(self):
        w = _worker()
        w._fps = 15.0
        w._engage_auto_low_fps(reason="fullscreen")
        assert w._low_fps_engaged_tier == "fullscreen"

    @pytest.mark.parametrize("fps", [11.9, 10.3, 9.0, 8.1, 8.0, 5.0, 1.0])
    def test_the_tier_no_longer_depends_on_the_fps_value(self, fps):
        """This is the regression. Every one of these tripped the critical
        gate; only those under 8.0 used to be filed as critical."""
        w = _worker()
        w._fps = fps
        w._engage_auto_low_fps(reason="critical")
        assert w._low_fps_engaged_tier == "critical", (
            f"{fps} fps tripped the critical gate but was filed as "
            f"{w._low_fps_engaged_tier}"
        )

    def test_the_field_rig_is_no_longer_filed_as_fullscreen(self):
        """10.3 fps is below the 12.0 critical gate but above the old
        12.0 - 4.0 = 8.0 margin. That gap is where he was stranded."""
        assert NE.GestureWorker._CRITICAL_FPS_THRESHOLD == 12.0
        margin = NE.GestureWorker._CRITICAL_FPS_ENGAGE_MARGIN
        old_cutoff = NE.GestureWorker._CRITICAL_FPS_THRESHOLD - margin
        assert 10.3 < NE.GestureWorker._CRITICAL_FPS_THRESHOLD
        assert not (10.3 < old_cutoff), "the stranding gap has moved"
        w = _worker()
        w._fps = 10.3
        w._engage_auto_low_fps(reason="critical")
        assert w._low_fps_engaged_tier == "critical"


class TestTheHardEscapeIsNowReachable:
    def test_the_hard_escape_is_gated_on_the_critical_tier(self):
        """Documents the coupling: filing the tier wrong switches off the
        only unconditional way out."""
        import inspect

        src = inspect.getsource(NE.GestureWorker._maybe_auto_toggle_low_fps)
        assert 'if tier == "critical":' in src

    def test_a_critical_engage_records_a_timestamp_for_it(self):
        w = _worker()
        w._fps = 10.3
        w._engage_auto_low_fps(reason="critical")
        assert w._low_fps_engaged_at is not None


class TestBothCallersStateTheirReason:
    def test_neither_call_site_relies_on_the_default(self):
        import inspect

        src = inspect.getsource(NE.GestureWorker._maybe_auto_toggle_low_fps)
        assert '_engage_auto_low_fps(reason="critical")' in src
        assert '_engage_auto_low_fps(reason="fullscreen")' in src
        assert "_engage_auto_low_fps()" not in src

    def test_the_tier_assignment_no_longer_reads_fps(self):
        import inspect
        import re

        src = inspect.getsource(NE.GestureWorker._engage_auto_low_fps)
        body = "\n".join(re.sub(r"#.*$", "", ln) for ln in src.splitlines())
        i = body.index("_low_fps_engaged_tier")
        assign = body[i:i + 220]
        assert "self._fps" not in assign
        assert "_CRITICAL_FPS_ENGAGE_MARGIN" not in assign
