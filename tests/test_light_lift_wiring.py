"""r24: the light lift must run in the pre-open window, and only there.

`light_policy` is pure and separately tested. This file tests the
wiring, which is where this codebase's defects actually live.

The load-bearing test is `TestItNeverTouchesTheLiveCapture`. Checkpoint
3.9 records that on the dad rig -- the exact hardware this feature
targets -- the first extra DirectShow property writes issued against a
live `ThreadedCvCapture` killed START with a native access violation,
with "DSHOW property Set racing the reader thread's `cap.read()` on the
same filter graph" as the leading hypothesis. 2.14 lists driver-side
gain compensation as deliberately not done for that reason.

The first draft of this feature ran on the live cap and argued it was
safe because it inherited the call site and gate of the EXPOSURE write
already there. That reasoning was wrong: what changed in r17 was the
NUMBER of Sets racing the reader, and the lift turns one into five.
These tests exist so that argument cannot be made again silently.
"""

import time

import numpy as np
import pytest

import hgr.app.integration.noop_engine as NE
from hgr.app.camera import light_policy as LP

W = NE.GestureWorker

WIN_ONLY = pytest.mark.skipif(NE.sys.platform != "win32",
                              reason="DirectShow path is Windows-only")


def _worker():
    w = W.__new__(W)
    w._note_driver_write = lambda name="Exposure": w._ledger.append(name)
    w._ledger = []
    return w


class FakeCap:
    """A throwaway cv2.VideoCapture: blocking reads, real responses.

    `direction=-1` models a driver whose Gamma runs the other way -- a
    real possibility the whole measure-and-revert design exists for.
    """

    def __init__(self, luma=10.0, direction=1, accept=True, latency_frames=0):
        self.luma = luma
        self.direction = direction
        self.accept = accept
        self.latency_frames = latency_frames
        self.sets = []
        self._pending = []
        self._n = 0

    def isOpened(self):
        return True

    def read(self):
        self._n += 1
        # Frames still in flight show the OLD picture for a while after
        # a property write -- the 500-800 ms DirectShow renegotiation.
        if self._pending and self._n >= self._pending[0][0]:
            self.luma = self._pending.pop(0)[1]
        v = int(max(0, min(255, self.luma)))
        rng = np.random.default_rng(v + 3)
        return True, rng.integers(max(0, v - 20), min(255, v + 21),
                                  (48, 64, 3), dtype=np.uint8)

    def set(self, prop, val):
        self.sets.append((prop, val))
        if not self.accept:
            return False
        import cv2
        delta = 0.0
        if prop == getattr(cv2, "CAP_PROP_GAMMA", -991):
            delta = (val - 165) * 0.35 * self.direction
            target = max(0.0, 10.0 + delta)
        elif prop == getattr(cv2, "CAP_PROP_BRIGHTNESS", -992):
            target = max(0.0, self.luma + val * 0.8 * self.direction)
        else:
            return True
        self._pending.append((self._n + self.latency_frames, target))
        return True

    def release(self):
        pass


FIELD_PROCAMP = {
    "Brightness": {"index": 0, "value": 0, "min": -64, "max": 64,
                   "default": 0, "flags": LP.FLAG_MANUAL},
    "Gamma": {"index": 5, "value": 165, "min": 100, "max": 500,
              "default": 100, "flags": LP.FLAG_MANUAL},
}


@pytest.fixture
def snap(monkeypatch):
    from hgr.app.camera import dshow_controls as DC
    monkeypatch.setattr(
        DC, "snapshot_all",
        lambda only_name=None: {
            "FULL HD 1080P Webcam": {"video_proc_amp": dict(FIELD_PROCAMP)}},
    )


# --------------------------------------------------------------------

class TestItNeverTouchesTheLiveCapture:
    """The rule that matters. See the module docstring."""

    def test_the_live_tuning_path_does_not_call_the_lift(self):
        import inspect
        import re

        src = inspect.getsource(W._apply_default_capture_tuning)
        body = "\n".join(re.sub(r"#.*$", "", ln) for ln in src.splitlines())
        assert "_recover_light_after_short_shutter(" not in body, (
            "the lift is being called on the LIVE engine capture -- "
            "checkpoint 3.9 forbids this and it crashed the dad rig"
        )
        assert "_lift_light_pre_open(" not in body

    def test_every_call_site_is_a_pre_open_window(self):
        """Each caller must own a throwaway cap, not the engine's."""
        import inspect
        import re

        callers = {"_lift_light_pre_open",
                   "_preflight_short_shutter_for_ffmpeg",
                   "_open_index_taking_the_light_window",
                   "_apply_perf_camera_path", "_open_camera",
                   "_maybe_lift_light_before_open"}
        src = inspect.getsource(NE)
        for m in re.finditer(r"self\._(?:recover_light_after_short_shutter|"
                             r"lift_light_pre_open)\(", src):
            line_no = src[:m.start()].count("\n") + 1
            owner = None
            for name in dir(W):
                fn = getattr(W, name, None)
                if not callable(fn) or not hasattr(fn, "__code__"):
                    continue
                try:
                    lines, start = inspect.getsourcelines(fn)
                except (OSError, TypeError):
                    continue
                if start <= line_no < start + len(lines):
                    if owner is None or start > owner[1]:
                        owner = (name, start)
            assert owner is not None and owner[0] in callers, (
                f"line {line_no}: lift invoked from {owner and owner[0]!r}, "
                f"which is not a known pre-open window"
            )

    def test_the_throwaway_is_opened_under_the_graph_lock(self):
        import inspect

        src = inspect.getsource(W._lift_light_pre_open)
        assert "DSHOW_GRAPH_LOCK" in src
        assert "cv2.CAP_DSHOW" in src

    def test_writes_go_through_the_graph_lock_helper(self):
        import inspect
        import re

        src = inspect.getsource(W._recover_light_after_short_shutter)
        body = "\n".join(re.sub(r"#.*$", "", ln) for ln in src.splitlines())
        assert "_with_graph_lock(cap.set" in body
        # A bare cap.set would bypass the bounded lock entirely.
        assert not re.search(r"(?<!_)\bcap\.set\(", body), body


class TestTheSampler:
    def test_it_measures_a_plain_capture(self):
        assert _worker()._sample_median_luma(FakeCap(luma=30.0)) == \
            pytest.approx(30.0, abs=3.0)

    def test_it_discards_stale_frames_first(self):
        """After a write, frames in flight still show the old picture."""
        cap = FakeCap(luma=10.0)
        cap._pending = [(3, 200.0)]      # picture changes at frame 3
        got = _worker()._sample_median_luma(cap, discard=2, samples=3)
        assert got > 150, (
            f"measured {got}: the sampler averaged in pre-write frames"
        )

    def test_a_dead_camera_returns_none_rather_than_hanging(self):
        class Dead:
            def read(self):
                return False, None
        t0 = time.monotonic()
        assert _worker()._sample_median_luma(Dead(), budget_s=0.25) is None
        assert time.monotonic() - t0 < 1.5

    def test_a_deadline_overrides_the_per_sample_budget(self):
        class Dead:
            def read(self):
                return False, None
        t0 = time.monotonic()
        _worker()._sample_median_luma(Dead(), budget_s=10.0,
                                      deadline=time.monotonic() + 0.2)
        assert time.monotonic() - t0 < 1.5

    def test_baseline_and_post_write_are_gathered_the_same_way(self):
        """An asymmetric baseline reads a stale bright frame and
        concludes the picture was never dark."""
        import inspect
        import re

        src = inspect.getsource(W._recover_light_after_short_shutter)
        body = "\n".join(re.sub(r"#.*$", "", ln) for ln in src.splitlines())
        calls = re.findall(r"_sample_median_luma\(([^)]*)\)", body)
        assert len(calls) == 2, calls
        for c in calls:
            assert "discard" not in c, (
                "one measurement overrides discard and the other does not"
            )


@WIN_ONLY
class TestTheLift:
    def test_a_dark_frame_gets_brighter(self, snap):
        cap = FakeCap(luma=10.0)
        _worker()._recover_light_after_short_shutter(cap, "FULL HD 1080P Webcam")
        assert cap.luma > 10.0 + LP.MIN_USEFUL_GAIN

    def test_a_backwards_gamma_driver_is_reverted(self, snap):
        import cv2
        cap = FakeCap(luma=10.0, direction=-1)
        _worker()._recover_light_after_short_shutter(cap, "FULL HD 1080P Webcam")
        gammas = [v for p, v in cap.sets
                  if p == getattr(cv2, "CAP_PROP_GAMMA", -991)]
        assert gammas and gammas[-1] == pytest.approx(165.0), gammas

    def test_a_bright_frame_is_left_alone(self, snap):
        cap = FakeCap(luma=150.0)
        _worker()._recover_light_after_short_shutter(cap, "FULL HD 1080P Webcam")
        assert cap.sets == []

    def test_a_qt_name_with_a_camera_suffix_matches(self, snap):
        cap = FakeCap(luma=10.0)
        _worker()._recover_light_after_short_shutter(
            cap, "FULL HD 1080P Webcam (Camera 0)")
        assert cap.luma > 10.0 + LP.MIN_USEFUL_GAIN

    def test_a_rejected_write_costs_nothing_extra(self, snap):
        """A False return is the one reliable 'it did not land' signal."""
        cap = FakeCap(luma=10.0, accept=False)
        _worker()._recover_light_after_short_shutter(cap, "FULL HD 1080P Webcam")
        import cv2
        gammas = [v for p, v in cap.sets
                  if p == getattr(cv2, "CAP_PROP_GAMMA", -991)]
        assert len(gammas) == 1, (
            f"reverted a write the driver never accepted: {cap.sets}"
        )

    def test_it_stays_inside_its_budget(self, snap):
        cap = FakeCap(luma=10.0, direction=-1)   # nothing ever improves
        t0 = time.monotonic()
        _worker()._recover_light_after_short_shutter(cap, "FULL HD 1080P Webcam")
        assert time.monotonic() - t0 < W._LIGHT_LIFT_TOTAL_BUDGET_S + 2.0

    def test_a_com_failure_is_survived(self, monkeypatch):
        from hgr.app.camera import dshow_controls as DC
        monkeypatch.setattr(DC, "snapshot_all", lambda only_name=None: (
            (_ for _ in ()).throw(RuntimeError("COM unavailable"))))
        cap = FakeCap(luma=10.0)
        _worker()._recover_light_after_short_shutter(cap, "FULL HD 1080P Webcam")
        assert cap.sets == []

    def test_the_kill_switch_stops_it(self, snap, monkeypatch):
        monkeypatch.setenv(W._LIGHT_LIFT_KILL_SWITCH, "0")
        cap = FakeCap(luma=10.0)
        _worker()._recover_light_after_short_shutter(cap, "FULL HD 1080P Webcam")
        assert cap.sets == []


@WIN_ONLY
class TestTheLedgerRecordsOnlyWhatWeKept:
    """STOP restores every property the ledger calls ours. Recording a
    reverted write would roll back a change the USER made in Synapse."""

    def test_a_kept_lift_is_recorded(self, snap):
        w = _worker()
        w._recover_light_after_short_shutter(FakeCap(luma=10.0),
                                             "FULL HD 1080P Webcam")
        assert "Gamma" in w._ledger

    def test_a_reverted_lift_is_not_recorded(self, snap):
        w = _worker()
        w._recover_light_after_short_shutter(FakeCap(luma=10.0, direction=-1),
                                             "FULL HD 1080P Webcam")
        assert w._ledger == [], (
            f"claimed ownership of properties it put back: {w._ledger}"
        )

    def test_a_rejected_write_is_not_recorded(self, snap):
        w = _worker()
        w._recover_light_after_short_shutter(FakeCap(luma=10.0, accept=False),
                                             "FULL HD 1080P Webcam")
        assert w._ledger == []


class TestDeviceMatching:
    def test_there_is_no_single_camera_fallback(self):
        """Every value in the block becomes a cap.set target, so another
        device's ranges must never be substituted."""
        snap = {"Some Other Webcam": {"video_proc_amp": dict(FIELD_PROCAMP)}}
        pa, how = W._procamp_for_device(snap, "Phone Camera (QR)")
        assert pa == {}, f"matched the wrong device: {how}"

    def test_an_empty_name_matches_nothing(self):
        snap = {"FULL HD 1080P Webcam": {"video_proc_amp": dict(FIELD_PROCAMP)}}
        assert W._procamp_for_device(snap, "")[0] == {}
        assert W._procamp_for_device(snap, None)[0] == {}

    @pytest.mark.parametrize("name", [
        "FULL HD 1080P Webcam",
        "full hd 1080p webcam",
        "FULL HD 1080P Webcam (Camera 0)",
        "FULL HD 1080P Webcam (Camera 12)",
    ])
    def test_the_real_device_matches_in_its_various_spellings(self, name):
        snap = {"FULL HD 1080P Webcam": {"video_proc_amp": dict(FIELD_PROCAMP)}}
        pa, how = W._procamp_for_device(snap, name)
        assert pa, how

    def test_an_empty_snapshot_is_not_fatal(self):
        assert W._procamp_for_device({}, "x")[0] == {}
        assert W._procamp_for_device(None, "x")[0] == {}


class TestTheHintGate:
    """Mirrors the r49 decision; a wrong answer means lifting a premium
    camera that never asked for a short shutter."""

    def _w(self, **cfg):
        w = W.__new__(W)
        w.config = type("C", (), dict(
            {"camera_force_short_shutter": False,
             "camera_force_short_shutter_user_chose": False}, **cfg))()
        return w

    def test_env_force_wins(self, monkeypatch):
        monkeypatch.setenv("HGR_FORCE_SHORT_SHUTTER", "1")
        assert self._w()._shutter_hint_applies("Anything") is True

    def test_env_off_wins(self, monkeypatch):
        monkeypatch.setenv("HGR_FORCE_SHORT_SHUTTER", "0")
        assert self._w(camera_force_short_shutter=True)._shutter_hint_applies(
            "x") is False

    def test_an_explicit_user_off_is_respected(self, monkeypatch):
        monkeypatch.delenv("HGR_FORCE_SHORT_SHUTTER", raising=False)
        w = self._w(camera_force_short_shutter=False,
                    camera_force_short_shutter_user_chose=True)
        assert w._shutter_hint_applies("FULL HD 1080P Webcam") is False

    def test_the_checkbox_on_is_respected(self, monkeypatch):
        monkeypatch.delenv("HGR_FORCE_SHORT_SHUTTER", raising=False)
        w = self._w(camera_force_short_shutter=True,
                    camera_force_short_shutter_user_chose=True)
        assert w._shutter_hint_applies("FULL HD 1080P Webcam") is True

    def test_the_classifier_alone_does_not_arm_it(self, monkeypatch):
        """A dark preview on first launch is the worst first impression
        this app can make, so a positive classification stays opt-in."""
        monkeypatch.delenv("HGR_FORCE_SHORT_SHUTTER", raising=False)
        monkeypatch.delenv("HGR_CLASSIFIER_AUTOSHUTTER", raising=False)
        assert self._w()._shutter_hint_applies("USB Camera") is False


class TestTheWiringExists:
    def test_the_ffmpeg_preflight_reuses_its_own_throwaway(self):
        """It already holds an open pre-open cap; a second open would be
        a second 500-800 ms device bind for nothing."""
        import inspect
        import re

        src = inspect.getsource(W._preflight_short_shutter_for_ffmpeg)
        body = "\n".join(re.sub(r"#.*$", "", ln) for ln in src.splitlines())
        assert "_lift_light_pre_open(index, device_name, cap=pre)" in body

    def test_every_reopen_takes_a_pre_open_window(self):
        """The field rig's real route is one of these: the ffmpeg memo
        skips the MJPG cascade, so no preflight runs and we land on an
        OpenCV cap. Routing ALL of them through one helper is what stops
        this from being "the branch we remembered" -- the first attempt
        covered one of four."""
        import inspect
        import re

        src = inspect.getsource(W._apply_perf_camera_path)
        body = "\n".join(re.sub(r"#.*$", "", ln) for ln in src.splitlines())
        assert "open_camera_by_index(" not in body, (
            "a re-open bypasses the light window"
        )
        assert body.count("_open_index_taking_the_light_window(") >= 4

    def test_the_helper_lifts_before_it_opens(self):
        import inspect
        import re

        src = inspect.getsource(W._open_index_taking_the_light_window)
        body = "\n".join(re.sub(r"#.*$", "", ln) for ln in src.splitlines())
        assert body.index("_lift_light_pre_open(") < \
            body.index("open_camera_by_index("), (
            "the lift must run BEFORE the capture exists"
        )
        assert "_shutter_hint_applies(" in body

    def test_the_helper_skips_an_unnamed_device(self):
        """One call site reaches it with '' by construction; without a
        name the lift can only open a capture and give up."""
        w = W.__new__(W)
        w.config = type("C", (), {"camera_scan_limit": 4})()
        w._shutter_hint_applies = lambda n: True
        called = []
        w._lift_light_pre_open = lambda *a, **k: called.append(a)
        import hgr.app.integration.noop_engine as mod
        real = mod.open_camera_by_index
        mod.open_camera_by_index = lambda *a, **k: (None, None)
        try:
            w._open_index_taking_the_light_window(0, "")
            assert called == []
            w._open_index_taking_the_light_window(0, "Real Cam")
            assert called == [(0, "Real Cam")]
        finally:
            mod.open_camera_by_index = real

    def test_the_ordinary_open_takes_one_too(self):
        import inspect
        import re

        src = inspect.getsource(W._open_camera)
        body = "\n".join(re.sub(r"#.*$", "", ln) for ln in src.splitlines())
        assert "_maybe_lift_light_before_open()" in body

    def test_the_resolver_uses_the_qt_only_enumeration(self):
        """The cv2 probe builds a DirectShow graph per device and can
        segfault on a bad third-party filter."""
        import inspect

        src = inspect.getsource(W._maybe_lift_light_before_open)
        assert "list_cameras_qt_only" in src
        assert "list_available_cameras" not in src

    def test_a_measured_device_is_not_lifted_twice(self):
        """The properties latch at the device, so repeating the lift on
        every mode swap costs a camera open and buys nothing."""
        w = W.__new__(W)
        w.config = type("C", (), {})()
        calls = []

        def measured(c, n):
            calls.append(n)
            return True          # we got a reading

        w._recover_light_after_short_shutter = measured
        w._lift_light_pre_open(0, "Cam A", cap=object())
        w._lift_light_pre_open(0, "Cam A", cap=object())
        w._lift_light_pre_open(1, "Cam B", cap=object())
        assert calls == ["Cam A", "Cam B"], calls

    def test_an_attempt_that_measured_nothing_does_not_burn_the_token(self):
        """r24 hotfix. The token used to be spent before the camera was
        even opened, so one transient (device busy mid-swap, no frames
        published yet) disabled the lift for the whole session -- on the
        exact rig the feature exists for."""
        w = W.__new__(W)
        w.config = type("C", (), {})()
        calls = []
        outcome = [False, False, True]     # two misses, then a reading

        def flaky(c, n):
            calls.append(n)
            return outcome[len(calls) - 1]

        w._recover_light_after_short_shutter = flaky
        for _ in range(4):
            w._lift_light_pre_open(0, "Cam A", cap=object())
        assert calls == ["Cam A"] * 3, (
            f"expected retries until a reading, then stop; got {calls}"
        )
