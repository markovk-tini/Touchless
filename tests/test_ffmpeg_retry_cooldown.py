"""r23: throttle ffmpeg re-attempts so a failing camera stops spawning
processes for the antivirus to prompt about.

The persisted two-strike memo (`ffmpeg_memo`) is deliberately slow: one
strike would permanently demote a camera that was merely busy for a
moment. But nothing throttled the attempts BETWEEN strikes, and
`_apply_perf_camera_path` reads "not currently on an ffmpeg capture" as
"never tried", so every Lite/GPU toggle re-ran the whole cascade. Worse,
`_engage_auto_low_fps` re-enters that path with no user action at all on
any rig sitting under `_CRITICAL_FPS_THRESHOLD` (12 fps) -- which
oscillates, and each pass is a fresh ffmpeg process. That is the field
rig's "5 norton ffmpeg access prompts".

The cooldown is in-memory and self-expiring, so unlike a session-wide
condemnation it can never permanently demote a good camera.
"""

import time

import pytest

import hgr.app.integration.noop_engine as NE


def _worker():
    w = NE.GestureWorker.__new__(NE.GestureWorker)
    w._r55_log = lambda *a, **k: None
    return w


CAM = "FULL HD 1080P Webcam"


class TestCooldownMechanics:
    def test_a_fresh_worker_has_no_cooldown(self):
        assert _worker()._ffmpeg_cooldown_active(CAM, 640, 480) is False

    def test_noting_a_failure_suppresses_the_next_attempt(self):
        w = _worker()
        w._ffmpeg_cooldown_note(CAM, 640, 480)
        assert w._ffmpeg_cooldown_active(CAM, 640, 480) is True

    def test_the_cooldown_is_per_device(self):
        w = _worker()
        w._ffmpeg_cooldown_note(CAM, 640, 480)
        assert w._ffmpeg_cooldown_active("Razer Kiyo Pro", 640, 480) is False

    def test_the_cooldown_is_per_resolution(self):
        """A camera with no 720p MJPG pin may still have a 480p one."""
        w = _worker()
        w._ffmpeg_cooldown_note(CAM, 1280, 720)
        assert w._ffmpeg_cooldown_active(CAM, 640, 480) is False

    def test_device_name_matching_ignores_case_and_padding(self):
        w = _worker()
        w._ffmpeg_cooldown_note("  FULL HD 1080p WEBCAM ", 640, 480)
        assert w._ffmpeg_cooldown_active(CAM, 640, 480) is True

    def test_it_expires_on_its_own(self, monkeypatch):
        w = _worker()
        w._ffmpeg_cooldown_note(CAM, 640, 480)
        base = time.monotonic()
        monkeypatch.setattr(
            NE.time, "monotonic",
            lambda: base + NE.GestureWorker._FFMPEG_RETRY_COOLDOWN_S + 1.0,
        )
        assert w._ffmpeg_cooldown_active(CAM, 640, 480) is False

    def test_an_expired_entry_is_dropped_not_left_to_accumulate(self, monkeypatch):
        w = _worker()
        w._ffmpeg_cooldown_note(CAM, 640, 480)
        base = time.monotonic()
        monkeypatch.setattr(
            NE.time, "monotonic",
            lambda: base + NE.GestureWorker._FFMPEG_RETRY_COOLDOWN_S + 1.0,
        )
        w._ffmpeg_cooldown_active(CAM, 640, 480)
        assert w._ffmpeg_retry_cooldown == {}

    def test_recovery_clears_it_so_a_freed_camera_comes_straight_back(self):
        w = _worker()
        w._ffmpeg_cooldown_note(CAM, 640, 480)
        w._ffmpeg_cooldown_clear()
        assert w._ffmpeg_cooldown_active(CAM, 640, 480) is False

    def test_clearing_an_empty_book_is_not_an_error(self):
        _worker()._ffmpeg_cooldown_clear()


class TestItThrottlesTheRealStorm:
    def test_a_burst_of_mode_toggles_attempts_ffmpeg_once(self, monkeypatch):
        """Ten Lite/GPU toggles in a row must not be ten ffmpeg spawns."""
        w = _worker()
        w.config = type("C", (), {"camera_ffmpeg_hard_failures": {}})()
        attempts = 0
        for _ in range(10):
            if not w._ffmpeg_cooldown_active(CAM, 640, 480):
                attempts += 1
                w._ffmpeg_cooldown_note(CAM, 640, 480)
        assert attempts == 1

    def test_after_the_cooldown_lapses_we_try_again_exactly_once_more(self, monkeypatch):
        w = _worker()
        w._ffmpeg_cooldown_note(CAM, 640, 480)
        base = time.monotonic()
        monkeypatch.setattr(
            NE.time, "monotonic",
            lambda: base + NE.GestureWorker._FFMPEG_RETRY_COOLDOWN_S + 1.0,
        )
        attempts = 0
        for _ in range(5):
            if not w._ffmpeg_cooldown_active(CAM, 640, 480):
                attempts += 1
                w._ffmpeg_cooldown_note(CAM, 640, 480)
        assert attempts == 1


class TestItDoesNotRegressThePersistedMemo:
    """PERFORMANCE_CHECKPOINT §2.11 / ffmpeg_memo.py:16-19.

    The two-strike rule exists so a momentarily-busy premium camera is
    never permanently demoted. The cooldown must stay a separate,
    self-expiring thing and must not become a one-strike memo.
    """

    def test_the_cooldown_never_writes_to_config(self):
        w = _worker()
        w.config = type("C", (), {"camera_ffmpeg_hard_failures": {}})()
        w._ffmpeg_cooldown_note(CAM, 640, 480)
        assert w.config.camera_ffmpeg_hard_failures == {}

    def test_the_two_strike_threshold_is_untouched(self):
        from hgr.app.camera import ffmpeg_memo

        assert ffmpeg_memo.DEFAULT_STRIKE_THRESHOLD == 2

    def test_a_silent_hang_still_is_not_a_strike_on_its_own(self):
        """The property the cooldown must not quietly undo."""
        from hgr.app.camera import ffmpeg_memo

        memo, changed = ffmpeg_memo.record_failure(
            {}, CAM, 640, 480, ffmpeg_memo.KIND_SILENT,
        )
        assert changed is False
        assert memo == {}

    def test_the_cooldown_is_shorter_than_a_typical_session(self):
        """It must forget by itself; that is what makes it safe."""
        assert 0.0 < NE.GestureWorker._FFMPEG_RETRY_COOLDOWN_S <= 300.0


class TestWiring:
    def test_the_skip_check_consults_the_cooldown(self):
        import inspect

        src = inspect.getsource(NE.GestureWorker._ffmpeg_memo_says_skip)
        assert "_ffmpeg_cooldown_active" in src

    def test_every_recorded_failure_starts_a_cooldown(self):
        import inspect

        src = inspect.getsource(NE.GestureWorker._ffmpeg_memo_record)
        assert "_ffmpeg_cooldown_note" in src
        # and it must come before the early `return` on "not a strike",
        # or the verdicts the memo ignores would keep re-spawning ffmpeg
        assert src.index("_ffmpeg_cooldown_note") < src.index("if not changed")

    def test_camera_recovery_clears_the_cooldown(self):
        import inspect

        src = inspect.getsource(NE.GestureWorker._attempt_camera_recovery)
        assert "_ffmpeg_cooldown_clear" in src
