"""Tests for the dynamic-gesture recorder + registry round-trip.

Covers:
  * Recorder state machine (idle → active → idle → ... → complete)
  * Three duration modes (1.5s, 3s, until-stopped)
  * Auto-stop fires at the right wall-clock time
  * 10-take target unlocks build_artifacts
  * Registry add_dynamic + load round-trip preserves data
  * Saved + reloaded registry can be turned into runtime templates

Tests use deterministic timestamps fed via `feed_frame(timestamp=...)`
so wall-clock isn't a flake source.
"""
from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

from hgr.custom_gestures.dynamic_recorder import (
    DEFAULT_TARGET_TAKES,
    DurationMode,
    DynamicGestureRecorder,
    RecorderState,
)
from hgr.custom_gestures.dynamic_classifier import (
    DynamicGestureClassifier,
    build_template_from_takes,
)
from hgr.custom_gestures.registry import (
    Action,
    CustomGesture,
    GestureRegistry,
)
from hgr.custom_gestures.dynamic_recording import (
    DynamicGestureTake,
    NUM_LANDMARKS,
)

# Reuse the synthetic-take helpers from the key-point selector
# tests — same gesture vocabulary across all the dynamic-gesture
# test suites.
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from test_dynamic_gesture_key_points import (
    _make_down_swipe_take,
    _base_hand_landmarks,
    _curl_fingers_except,
)


def _stream_take_into_recorder(
    recorder: DynamicGestureRecorder,
    take: DynamicGestureTake,
    *,
    palm_scale: float = 1.0,
    start_t: float = 0.0,
    dt: float = 1.0 / 30.0,
) -> None:
    """Feed a synthetic take's frames into a recorder, simulating a
    30-fps live capture. begin_take() must already have been called."""
    t = start_t
    for f in range(take.num_frames):
        recorder.feed_frame(take.landmarks[f], palm_scale, timestamp=t)
        t += dt


class RecorderStateMachineTests(unittest.TestCase):

    def test_initial_state_is_idle(self) -> None:
        rec = DynamicGestureRecorder()
        self.assertEqual(rec.state, RecorderState.IDLE)
        self.assertEqual(rec.completed_takes, 0)

    def test_begin_take_transitions_to_active(self) -> None:
        rec = DynamicGestureRecorder()
        self.assertTrue(rec.begin_take())
        self.assertEqual(rec.state, RecorderState.ACTIVE)
        # Calling begin_take while ACTIVE is a no-op.
        self.assertFalse(rec.begin_take())

    def test_end_take_freezes_and_returns_to_idle(self) -> None:
        rec = DynamicGestureRecorder(DurationMode.UNTIL_STOPPED)
        rec.begin_take()
        # Feed 5 frames at fixed timestamps so duration is deterministic.
        for f in range(5):
            lm = _base_hand_landmarks()
            rec.feed_frame(lm, palm_scale=1.0, timestamp=float(f) * 0.05)
        take = rec.end_take()
        self.assertIsNotNone(take)
        self.assertEqual(take.num_frames, 5)
        self.assertEqual(rec.state, RecorderState.IDLE)
        self.assertEqual(rec.completed_takes, 1)

    def test_fixed_short_auto_stops_after_one_point_five_seconds(self) -> None:
        rec = DynamicGestureRecorder(DurationMode.FIXED_SHORT)
        rec.begin_take()
        # Feed frames with timestamps spanning past the 1.5s cap. The
        # very first frame whose elapsed >= 1.5 should trigger auto-stop.
        for f in range(60):
            t = f * 0.04  # 25 fps → 60 frames = 2.4 s, well past 1.5s
            lm = _base_hand_landmarks()
            rec.feed_frame(lm, palm_scale=1.0, timestamp=t)
            if rec.state == RecorderState.IDLE:
                break
        self.assertEqual(rec.state, RecorderState.IDLE)
        self.assertEqual(rec.completed_takes, 1)
        # Final take should have closed near the 1.5s mark.
        take = rec.takes[0]
        self.assertGreaterEqual(take.raw_duration_seconds, 1.4)
        self.assertLessEqual(take.raw_duration_seconds, 1.7)

    def test_fixed_long_auto_stops_after_three_seconds(self) -> None:
        rec = DynamicGestureRecorder(DurationMode.FIXED_LONG)
        rec.begin_take()
        for f in range(120):
            t = f * 0.04  # 25 fps for 4.8 s
            rec.feed_frame(_base_hand_landmarks(), palm_scale=1.0, timestamp=t)
            if rec.state == RecorderState.IDLE:
                break
        take = rec.takes[0]
        self.assertGreaterEqual(take.raw_duration_seconds, 2.9)
        self.assertLessEqual(take.raw_duration_seconds, 3.2)

    def test_until_stopped_never_auto_stops(self) -> None:
        rec = DynamicGestureRecorder(DurationMode.UNTIL_STOPPED)
        rec.begin_take()
        # Feed 10 seconds of frames; recorder should still be active.
        for f in range(300):
            rec.feed_frame(_base_hand_landmarks(), palm_scale=1.0, timestamp=f * 0.034)
        self.assertEqual(rec.state, RecorderState.ACTIVE)
        rec.end_take()
        self.assertEqual(rec.state, RecorderState.IDLE)

    def test_complete_state_reached_after_target_takes(self) -> None:
        rec = DynamicGestureRecorder(DurationMode.UNTIL_STOPPED, target_takes=3)
        for _ in range(3):
            rec.begin_take()
            rec.feed_frame(_base_hand_landmarks(), palm_scale=1.0, timestamp=0.0)
            rec.end_take()
        self.assertEqual(rec.state, RecorderState.COMPLETE)

    def test_discard_active_does_not_save_take(self) -> None:
        rec = DynamicGestureRecorder(DurationMode.UNTIL_STOPPED)
        rec.begin_take()
        rec.feed_frame(_base_hand_landmarks(), palm_scale=1.0, timestamp=0.0)
        rec.discard_active()
        self.assertEqual(rec.state, RecorderState.IDLE)
        self.assertEqual(rec.completed_takes, 0)

    def test_discard_take_removes_completed_recording(self) -> None:
        rec = DynamicGestureRecorder(DurationMode.UNTIL_STOPPED, target_takes=2)
        for _ in range(2):
            rec.begin_take()
            rec.feed_frame(_base_hand_landmarks(), palm_scale=1.0, timestamp=0.0)
            rec.end_take()
        self.assertEqual(rec.state, RecorderState.COMPLETE)
        self.assertTrue(rec.discard_take(0))
        # Should drop back to IDLE since we're below target now.
        self.assertEqual(rec.state, RecorderState.IDLE)
        self.assertEqual(rec.completed_takes, 1)

    def test_feed_frame_rejected_when_not_active(self) -> None:
        rec = DynamicGestureRecorder()
        ok = rec.feed_frame(_base_hand_landmarks(), palm_scale=1.0, timestamp=0.0)
        self.assertFalse(ok)


class RecorderArtifactsTests(unittest.TestCase):

    def test_build_artifacts_produces_runnable_template(self) -> None:
        rec = DynamicGestureRecorder(DurationMode.UNTIL_STOPPED, target_takes=10)
        for _ in range(10):
            rec.begin_take()
            _stream_take_into_recorder(rec, _make_down_swipe_take())
            rec.end_take()
        artifacts = rec.build_artifacts(gesture_name="down_swipe")
        # Key-point selection should have flagged at least the index
        # finger landmarks as essential.
        self.assertIn(8, artifacts.key_points.indices)
        # The runtime template has one trajectory per take.
        self.assertEqual(len(artifacts.template.sample_trajectories), 10)
        # Template is usable by the classifier — instantiate and confirm
        # the down-swipe takes themselves match against it.
        classifier = DynamicGestureClassifier([artifacts.template])
        # Stream a fresh take through; the classifier should detect.
        from test_dynamic_gesture_classifier import _stream_take_through
        events = _stream_take_through(classifier, _make_down_swipe_take())
        matches = [m for _, m in events if m is not None]
        self.assertTrue(matches, "recorder→template→classifier pipeline should match")
        self.assertEqual(matches[0].gesture_name, "down_swipe")


class RegistryRoundTripTests(unittest.TestCase):

    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="dyn_gesture_registry_"))
        self.registry_path = self.tmp / "custom_gestures.json"

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _record_swipe(self, take_count: int = 10):
        rec = DynamicGestureRecorder(DurationMode.UNTIL_STOPPED, target_takes=take_count)
        for _ in range(take_count):
            rec.begin_take()
            _stream_take_into_recorder(rec, _make_down_swipe_take())
            rec.end_take()
        return rec.build_artifacts(gesture_name="down_swipe")

    def test_add_dynamic_persists_and_reloads(self) -> None:
        artifacts = self._record_swipe()
        registry = GestureRegistry(self.registry_path)
        registry.add_dynamic(
            name="down_swipe",
            key_point_indices=artifacts.key_points.indices,
            sample_trajectories=artifacts.template.sample_trajectories,
            action=Action(kind="hotkey", payload={"keys": ["ctrl", "d"]}),
            description="swipe down to do thing",
            handedness="Right",
            duration_mode="until_stopped",
        )
        registry.save()
        # Reload from scratch in a separate registry instance.
        reloaded = GestureRegistry(self.registry_path)
        reloaded.load()
        gestures = [g for g in reloaded._gestures.values()]  # noqa: SLF001 — test
        self.assertEqual(len(gestures), 1)
        g = gestures[0]
        self.assertEqual(g.name, "down_swipe")
        self.assertEqual(g.kind, "dynamic")
        self.assertEqual(g.duration_mode, "until_stopped")
        self.assertEqual(g.handedness, "Right")
        self.assertEqual(
            list(g.key_point_indices),
            list(artifacts.key_points.indices),
        )
        # Sample trajectories should round-trip with same shape.
        self.assertEqual(
            len(g.sample_trajectories),
            len(artifacts.template.sample_trajectories),
        )
        self.assertEqual(g.action.kind, "hotkey")
        self.assertEqual(g.action.payload["keys"], ["ctrl", "d"])

    def test_static_gestures_unaffected_by_dynamic_fields(self) -> None:
        # A v1-format JSON (no kind / no dynamic fields) must still
        # load cleanly with kind="static". Smoke test the backward
        # compat by writing a hand-built record.
        legacy_blob = {
            "schema_version": 1,
            "gestures": [
                {
                    "name": "old_static",
                    "description": "",
                    "created_at": "2026-01-01T00:00:00+00:00",
                    "handedness": "Right",
                    "image_filename": "",
                    "action": {"kind": "noop", "payload": {}},
                    "samples": [],
                }
            ],
        }
        self.registry_path.write_text(json.dumps(legacy_blob), encoding="utf-8")
        registry = GestureRegistry(self.registry_path)
        registry.load()
        gestures = list(registry._gestures.values())  # noqa: SLF001 — test
        self.assertEqual(len(gestures), 1)
        g = gestures[0]
        self.assertEqual(g.kind, "static")
        self.assertEqual(g.key_point_indices, [])
        self.assertEqual(g.sample_trajectories, [])

    def test_dynamic_gesture_matches_after_reload(self) -> None:
        artifacts = self._record_swipe()
        registry = GestureRegistry(self.registry_path)
        registry.add_dynamic(
            name="down_swipe",
            key_point_indices=artifacts.key_points.indices,
            sample_trajectories=artifacts.template.sample_trajectories,
            action=Action(kind="noop", payload={}),
            handedness="Right",
            duration_mode="until_stopped",
        )
        registry.save()
        # Reload and reconstruct a runtime template from the stored data.
        reloaded = GestureRegistry(self.registry_path)
        reloaded.load()
        g = list(reloaded._gestures.values())[0]  # noqa: SLF001
        # Build a template from the reloaded data — mirrors what the
        # engine will do at startup once the hook is wired.
        template = _template_from_registry_entry(g)
        classifier = DynamicGestureClassifier([template])
        from test_dynamic_gesture_classifier import _stream_take_through
        events = _stream_take_through(classifier, _make_down_swipe_take())
        matches = [m for _, m in events if m is not None]
        self.assertTrue(
            matches,
            msg="reloaded dynamic gesture should still match",
        )


def _template_from_registry_entry(g: CustomGesture):
    """Helper: turn a CustomGesture (kind=dynamic) into a runtime
    DynamicGestureTemplate. This logic will live in the engine-side
    integration code once we wire it in; for now we keep it here so
    the registry round-trip test can exercise it."""
    from hgr.custom_gestures.dynamic_classifier import DynamicGestureTemplate
    traj_arrays = [np.asarray(t, dtype=np.float32) for t in g.sample_trajectories]
    return DynamicGestureTemplate(
        name=g.name,
        key_point_indices=list(g.key_point_indices),
        sample_trajectories=traj_arrays,
    )


if __name__ == "__main__":
    unittest.main()
