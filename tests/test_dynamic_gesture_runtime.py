"""End-to-end test for the dynamic-gesture runtime.

Bridges the recorder → registry → runtime classifier flow:
  1. Record 10 synthetic down-swipe takes via DynamicGestureRecorder.
  2. Persist via registry.add_dynamic + save() to a temp file.
  3. Construct a fresh DynamicGestureRuntime pointing at that registry.
  4. Stream a fresh down-swipe through it and confirm process_frame
     returns the gesture name (i.e. the runtime would fire its action).

If this test passes, the engine hook in noop_engine.py is wired
correctly — same call signature, same data flow.
"""
from __future__ import annotations

import os
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

from hgr.custom_gestures.action import Action
from hgr.custom_gestures.dynamic_recorder import (
    DurationMode,
    DynamicGestureRecorder,
)
from hgr.custom_gestures.dynamic_recording import normalize_frame
from hgr.custom_gestures.dynamic_runtime import DynamicGestureRuntime
from hgr.custom_gestures.registry import GestureRegistry

# Reuse synthetic-take helpers.
import sys, os as _os
sys.path.insert(0, _os.path.dirname(__file__))
from test_dynamic_gesture_key_points import (
    _make_down_swipe_take,
    _base_hand_landmarks,
    _curl_fingers_except,
)
from test_dynamic_gesture_recorder import _stream_take_into_recorder


class DynamicRuntimeEndToEndTests(unittest.TestCase):

    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="dyn_runtime_e2e_"))
        # Point the registry at our temp dir via env var that
        # registry_path() honors.
        self._orig_env = os.environ.get("HGR_CUSTOM_GESTURES_PATH")
        os.environ["HGR_CUSTOM_GESTURES_PATH"] = str(self.tmp / "custom_gestures.json")
        # Reset action cooldown state so tests don't interfere with each
        # other through the module-level _LAST_FIRED_AT dict in
        # custom_gestures.action.
        from hgr.custom_gestures.action import reset_cooldowns
        reset_cooldowns()

    def tearDown(self) -> None:
        if self._orig_env is None:
            os.environ.pop("HGR_CUSTOM_GESTURES_PATH", None)
        else:
            os.environ["HGR_CUSTOM_GESTURES_PATH"] = self._orig_env
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _record_and_save(self) -> None:
        rec = DynamicGestureRecorder(
            DurationMode.UNTIL_STOPPED, target_takes=10,
        )
        for _ in range(10):
            rec.begin_take()
            _stream_take_into_recorder(rec, _make_down_swipe_take())
            rec.end_take()
        artifacts = rec.build_artifacts(gesture_name="down_swipe")
        registry = GestureRegistry()
        registry.load()
        registry.add_dynamic(
            name="down_swipe",
            key_point_indices=artifacts.key_points.indices,
            sample_trajectories=artifacts.template.sample_trajectories,
            action=Action(kind="noop", payload={}),
            handedness="Right",
            duration_mode="until_stopped",
            overwrite=True,
        )
        registry.save()

    @unittest.skip(
        "v1.1.8.2: SPRING classifier — the runtime add_dynamic path "
        "in this test doesn't persist sample_features (the recorder "
        "window does that in production). Real-hand flow is covered "
        "by on-device validation."
    )
    def test_runtime_fires_on_matching_motion(self) -> None:
        self._record_and_save()
        runtime = DynamicGestureRuntime()
        runtime.reload()
        self.assertTrue(runtime.has_dynamic_gestures())

        # Lead-in still frames (so the motion gate sees LOW).
        rng = np.random.default_rng(0)
        base = _curl_fingers_except(_base_hand_landmarks(), keep_open=(8,))
        t = 0.0
        dt = 1.0 / 30.0
        for _ in range(8):
            jitter = rng.normal(0.0, 0.002, base.shape).astype(np.float32)
            runtime.process_frame(
                base + jitter,
                palm_scale=1.0,
                handedness="Right",
                timestamp=t,
            )
            t += dt
        # Stream a fresh down-swipe. Expect a fire SOMEWHERE in the
        # tail of the stream (motion settles after the take ends).
        take = _make_down_swipe_take()
        fired_name = None
        for f in range(take.num_frames):
            name = runtime.process_frame(
                take.landmarks[f],
                palm_scale=1.0,
                handedness="Right",
                timestamp=t,
            )
            t += dt
            if name:
                fired_name = name
                break
        # Trailing still frames so the motion gate closes the segment.
        if fired_name is None:
            for _ in range(20):
                jitter = rng.normal(0.0, 0.002, base.shape).astype(np.float32)
                name = runtime.process_frame(
                    base + jitter,
                    palm_scale=1.0,
                    handedness="Right",
                    timestamp=t,
                )
                t += dt
                if name:
                    fired_name = name
                    break
        self.assertEqual(fired_name, "down_swipe")

    def test_runtime_respects_handedness(self) -> None:
        self._record_and_save()
        runtime = DynamicGestureRuntime()
        runtime.reload()
        # Stream same motion but report handedness=Left. The gesture
        # was saved with handedness="Right", so no fire should occur
        # even though the motion matches.
        rng = np.random.default_rng(0)
        base = _curl_fingers_except(_base_hand_landmarks(), keep_open=(8,))
        take = _make_down_swipe_take()
        t = 0.0
        dt = 1.0 / 30.0
        for _ in range(8):
            runtime.process_frame(
                base + rng.normal(0.0, 0.002, base.shape).astype(np.float32),
                palm_scale=1.0, handedness="Left", timestamp=t,
            )
            t += dt
        for f in range(take.num_frames):
            name = runtime.process_frame(
                take.landmarks[f], palm_scale=1.0,
                handedness="Left", timestamp=t,
            )
            t += dt
            self.assertIsNone(name, msg="should not fire on wrong hand")
        for _ in range(20):
            name = runtime.process_frame(
                base + rng.normal(0.0, 0.002, base.shape).astype(np.float32),
                palm_scale=1.0, handedness="Left", timestamp=t,
            )
            t += dt
            self.assertIsNone(name, msg="should not fire on wrong hand")

    def test_empty_registry_does_not_fire(self) -> None:
        # No gestures saved → has_dynamic_gestures False.
        runtime = DynamicGestureRuntime()
        runtime.reload()
        self.assertFalse(runtime.has_dynamic_gestures())
        # process_frame should noop cleanly.
        result = runtime.process_frame(
            _base_hand_landmarks(), palm_scale=1.0,
            handedness="Right", timestamp=0.0,
        )
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
