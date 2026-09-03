"""Tests for pose-sequence custom gestures."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from hgr.custom_gestures.pose_sequence_runtime import PoseSequenceRuntime
from hgr.custom_gestures.registry import (
    Action,
    GestureRegistry,
    GestureSample,
    PoseSequenceStep,
    _FEATURE_VECTOR_LEN,
)


def _fake_sample(seed: int) -> GestureSample:
    rng = np.random.default_rng(seed)
    feats = rng.standard_normal(_FEATURE_VECTOR_LEN).astype(np.float32) * 0.01
    return GestureSample(features=feats.tolist())


class PoseSequenceRegistryTests(unittest.TestCase):
    def test_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "g.json"
            reg = GestureRegistry(path)
            steps = [
                PoseSequenceStep("3", [_fake_sample(1), _fake_sample(2)]),
                PoseSequenceStep("2", [_fake_sample(3), _fake_sample(4)]),
                PoseSequenceStep("1", [_fake_sample(5), _fake_sample(6)]),
            ]
            reg.add_pose_sequence(
                "countdown",
                steps,
                Action(kind="noop", payload={"cooldown_s": 1.0}),
                handedness="Right",
                dwell_ms=200,
                max_hold_ms=500,
                max_gap_ms=900,
            )
            reg.save()
            reg2 = GestureRegistry(path)
            reg2.load()
            g = reg2.get("countdown")
            self.assertIsNotNone(g)
            assert g is not None
            self.assertEqual(g.kind, "pose_sequence")
            self.assertEqual(len(g.pose_sequence_steps), 3)
            self.assertEqual(g.pose_sequence_steps[0].name, "3")
            self.assertEqual(g.pose_sequence_dwell_ms, 200)
            self.assertEqual(g.pose_sequence_max_hold_ms, 500)
            self.assertEqual(g.pose_sequence_max_gap_ms, 900)
            self.assertEqual(g.handedness, "Right")

    def test_replace_metadata_preserves_kind_and_steps(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "g.json"
            reg = GestureRegistry(path)
            steps = [
                PoseSequenceStep("a", [_fake_sample(1)]),
                PoseSequenceStep("b", [_fake_sample(2)]),
            ]
            reg.add_pose_sequence(
                "seq",
                steps,
                Action(kind="noop", payload={"cooldown_s": 1.0}),
                description="old",
            )
            updated = reg.replace_metadata(
                "seq",
                name="seq_renamed",
                action=Action(kind="hotkey", payload={"keys": ["ctrl", "s"]}),
                description="new desc",
            )
            self.assertEqual(updated.name, "seq_renamed")
            self.assertEqual(updated.kind, "pose_sequence")
            self.assertEqual(updated.description, "new desc")
            self.assertEqual(updated.action.kind, "hotkey")
            self.assertEqual(len(updated.pose_sequence_steps), 2)
            self.assertIsNone(reg.get("seq"))
            self.assertIsNotNone(reg.get("seq_renamed"))


class PoseSequenceRuntimeTests(unittest.TestCase):
    def _make_ab_runtime(
        self,
        path: Path,
        *,
        dwell_ms: int = 150,
        max_hold_ms: int = 400,
        max_gap_ms: int = 900,
    ) -> PoseSequenceRuntime:
        reg = GestureRegistry(path)
        feats_a = np.zeros(_FEATURE_VECTOR_LEN, dtype=np.float32)
        feats_b = np.zeros(_FEATURE_VECTOR_LEN, dtype=np.float32)
        feats_b[0] = 5.0
        reg.add_pose_sequence(
            "ab",
            [
                PoseSequenceStep(
                    "a",
                    [GestureSample(features=feats_a.tolist()) for _ in range(3)],
                ),
                PoseSequenceStep(
                    "b",
                    [GestureSample(features=feats_b.tolist()) for _ in range(3)],
                ),
            ],
            Action(kind="noop", payload={"cooldown_s": 0.05}),
            dwell_ms=dwell_ms,
            max_hold_ms=max_hold_ms,
            max_gap_ms=max_gap_ms,
        )
        reg.save()
        rt = PoseSequenceRuntime(match_threshold=0.78)
        rt.reload()
        return rt

    def test_reload_finds_sequences(self) -> None:
        import os

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "g.json"
            old = os.environ.get("HGR_CUSTOM_GESTURES_PATH")
            os.environ["HGR_CUSTOM_GESTURES_PATH"] = str(path)
            try:
                rt = self._make_ab_runtime(path)
                self.assertTrue(rt.has_sequences())
                self.assertIn("ab", rt._states)
            finally:
                if old is None:
                    os.environ.pop("HGR_CUSTOM_GESTURES_PATH", None)
                else:
                    os.environ["HGR_CUSTOM_GESTURES_PATH"] = old

    def test_dwell_release_advances_and_fires(self) -> None:
        import os

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "g.json"
            old = os.environ.get("HGR_CUSTOM_GESTURES_PATH")
            os.environ["HGR_CUSTOM_GESTURES_PATH"] = str(path)
            try:
                rt = self._make_ab_runtime(path, dwell_ms=150, max_hold_ms=500)
                self.assertTrue(rt.has_sequences())
                lm = np.zeros((21, 3), dtype=np.float32)
                feats_a = np.zeros(_FEATURE_VECTOR_LEN, dtype=np.float32)
                feats_b = np.zeros(_FEATURE_VECTOR_LEN, dtype=np.float32)
                feats_b[0] = 5.0
                t = 1.0
                with mock.patch(
                    "hgr.custom_gestures.pose_sequence_runtime.normalize_landmarks",
                    side_effect=lambda _lm: feats_a.copy(),
                ), mock.patch(
                    "hgr.custom_gestures.pose_sequence_runtime.fire_once",
                ):
                    for _ in range(8):
                        rt.process_landmarks(lm, timestamp=t, dispatch=True)
                        t += 0.05
                    # Still holding pose A after dwell — not advanced yet.
                    self.assertEqual(rt._states["ab"].step_index, 0)
                    self.assertTrue(rt._states["ab"].dwell_met)
                    # Release A → advance to B.
                    with mock.patch(
                        "hgr.custom_gestures.pose_sequence_runtime.GestureClassifier.classify_raw",
                        return_value=None,
                    ):
                        rt.process_landmarks(lm, timestamp=t, dispatch=True)
                        t += 0.05
                    self.assertEqual(rt._states["ab"].step_index, 1)

                fired = None
                with mock.patch(
                    "hgr.custom_gestures.pose_sequence_runtime.normalize_landmarks",
                    side_effect=lambda _lm: feats_b.copy(),
                ), mock.patch(
                    "hgr.custom_gestures.pose_sequence_runtime.fire_once",
                ) as fire:
                    for _ in range(8):
                        name = rt.process_landmarks(lm, timestamp=t, dispatch=True)
                        if name:
                            fired = name
                        t += 0.05
                    self.assertEqual(fired, "ab")
                    fire.assert_called()
            finally:
                if old is None:
                    os.environ.pop("HGR_CUSTOM_GESTURES_PATH", None)
                else:
                    os.environ["HGR_CUSTOM_GESTURES_PATH"] = old

    def test_max_hold_resets(self) -> None:
        import os

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "g.json"
            old = os.environ.get("HGR_CUSTOM_GESTURES_PATH")
            os.environ["HGR_CUSTOM_GESTURES_PATH"] = str(path)
            try:
                rt = self._make_ab_runtime(
                    path, dwell_ms=100, max_hold_ms=200, max_gap_ms=900,
                )
                lm = np.zeros((21, 3), dtype=np.float32)
                feats_a = np.zeros(_FEATURE_VECTOR_LEN, dtype=np.float32)
                t = 1.0
                with mock.patch(
                    "hgr.custom_gestures.pose_sequence_runtime.normalize_landmarks",
                    side_effect=lambda _lm: feats_a.copy(),
                ), mock.patch(
                    "hgr.custom_gestures.pose_sequence_runtime.fire_once",
                ):
                    saw_dwell_met = False
                    saw_reset_after = False
                    for _ in range(12):
                        rt.process_landmarks(lm, timestamp=t, dispatch=True)
                        st = rt._states["ab"]
                        if st.dwell_met:
                            saw_dwell_met = True
                        if saw_dwell_met and st.dwell_started_at is None:
                            saw_reset_after = True
                        t += 0.05
                    self.assertTrue(saw_dwell_met)
                    self.assertTrue(saw_reset_after)
                    self.assertEqual(rt._states["ab"].step_index, 0)
            finally:
                if old is None:
                    os.environ.pop("HGR_CUSTOM_GESTURES_PATH", None)
                else:
                    os.environ["HGR_CUSTOM_GESTURES_PATH"] = old

    def test_max_gap_resets_after_advance(self) -> None:
        import os

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "g.json"
            old = os.environ.get("HGR_CUSTOM_GESTURES_PATH")
            os.environ["HGR_CUSTOM_GESTURES_PATH"] = str(path)
            try:
                rt = self._make_ab_runtime(
                    path, dwell_ms=100, max_hold_ms=400, max_gap_ms=200,
                )
                lm = np.zeros((21, 3), dtype=np.float32)
                feats_a = np.zeros(_FEATURE_VECTOR_LEN, dtype=np.float32)
                t = 1.0
                with mock.patch(
                    "hgr.custom_gestures.pose_sequence_runtime.normalize_landmarks",
                    side_effect=lambda _lm: feats_a.copy(),
                ), mock.patch(
                    "hgr.custom_gestures.pose_sequence_runtime.fire_once",
                ):
                    for _ in range(5):
                        rt.process_landmarks(lm, timestamp=t, dispatch=True)
                        t += 0.05
                    self.assertTrue(rt._states["ab"].dwell_met)
                    with mock.patch(
                        "hgr.custom_gestures.pose_sequence_runtime.GestureClassifier.classify_raw",
                        return_value=None,
                    ):
                        rt.process_landmarks(lm, timestamp=t, dispatch=True)
                        t += 0.05
                        self.assertEqual(rt._states["ab"].step_index, 1)
                        # Wait past max gap without showing pose B.
                        t += 0.25
                        rt.process_landmarks(lm, timestamp=t, dispatch=True)
                    self.assertEqual(rt._states["ab"].step_index, 0)
            finally:
                if old is None:
                    os.environ.pop("HGR_CUSTOM_GESTURES_PATH", None)
                else:
                    os.environ["HGR_CUSTOM_GESTURES_PATH"] = old


def _synthetic_hand(*, mode: str) -> np.ndarray:
    """Minimal valid (21,3) hand with clearly distinct modes."""
    lm = np.zeros((21, 3), dtype=np.float32)
    lm[0] = (0.0, 0.0, 0.0)
    lm[9] = (0.0, -0.2, 0.0)
    # Default: lightly curled fist-ish.
    for tip, mcp, x in (
        (4, 2, -0.08),
        (8, 5, -0.04),
        (12, 9, 0.0),
        (16, 13, 0.04),
        (20, 17, 0.08),
    ):
        lm[mcp] = (x, -0.12, 0.0)
        lm[mcp + 1] = (x, -0.10, 0.0)
        lm[mcp + 2] = (x, -0.08, 0.0) if tip - mcp >= 3 else lm[mcp + 1]
        lm[tip] = (x, -0.06, 0.0)
    lm[1] = (-0.05, -0.04, 0.0)
    lm[3] = (-0.07, -0.05, 0.0)

    if mode == "open":
        for tip, x in ((4, -0.12), (8, -0.06), (12, 0.0), (16, 0.06), (20, 0.12)):
            lm[tip] = (x, -0.42, 0.0)
    elif mode == "peace":
        lm[8] = (-0.05, -0.40, 0.0)
        lm[12] = (0.05, -0.40, 0.0)
        lm[16] = (0.04, -0.07, 0.0)
        lm[20] = (0.08, -0.06, 0.0)
        lm[4] = (-0.10, -0.08, 0.0)
    elif mode == "one":
        lm[8] = (0.0, -0.42, 0.0)
        lm[12] = (0.02, -0.07, 0.0)
        lm[16] = (0.05, -0.06, 0.0)
        lm[20] = (0.08, -0.05, 0.0)
        lm[4] = (-0.08, -0.07, 0.0)
    else:
        raise ValueError(mode)
    return lm


def _hold_pose_frames(
    pose_lm: np.ndarray,
    *,
    t0: float,
    hold_s: float,
    fps: float = 30.0,
    handedness: str = "Right",
    jitter: float = 0.0015,
):
    from hgr.custom_gestures.pose_sequence_analysis import SequenceFrame

    n = max(5, int(round(hold_s * fps)))
    out = []
    rng = np.random.default_rng(0)
    for i in range(n):
        lm = pose_lm + rng.normal(0.0, jitter, size=pose_lm.shape).astype(np.float32)
        out.append(
            SequenceFrame(
                t=t0 + i / fps,
                landmarks=lm,
                handedness=handedness,
            )
        )
    return out


def _transition_frames(
    a: np.ndarray,
    b: np.ndarray,
    *,
    t0: float,
    n: int = 4,
    fps: float = 30.0,
    handedness: str = "Right",
):
    from hgr.custom_gestures.pose_sequence_analysis import SequenceFrame

    out = []
    for i in range(n):
        alpha = (i + 1) / (n + 1)
        lm = (1.0 - alpha) * a + alpha * b
        out.append(
            SequenceFrame(
                t=t0 + i / fps,
                landmarks=lm.astype(np.float32),
                handedness=handedness,
            )
        )
    return out


class PoseSequenceAnalysisTests(unittest.TestCase):
    def test_learns_order_holds_gaps_and_hand(self) -> None:
        from hgr.custom_gestures.pose_sequence_analysis import analyze_pose_sequence

        p3 = _synthetic_hand(mode="open")
        p2 = _synthetic_hand(mode="peace")
        p1 = _synthetic_hand(mode="one")

        frames = []
        t = 0.0
        poses_holds = ((p3, 0.40), (p2, 0.35), (p1, 0.45))
        for i, (pose, hold) in enumerate(poses_holds):
            seg = _hold_pose_frames(pose, t0=t, hold_s=hold, handedness="Right")
            frames.extend(seg)
            t = frames[-1].t + (1.0 / 30.0)
            if i < len(poses_holds) - 1:
                nxt = poses_holds[i + 1][0]
                trans = _transition_frames(pose, nxt, t0=t, n=5)
                frames.extend(trans)
                t = frames[-1].t + (1.0 / 30.0)

        result = analyze_pose_sequence(
            frames, expected_steps=3, step_names=("Pose 1", "Pose 2", "Pose 3"),
        )
        self.assertEqual(result.step_names, ["Pose 1", "Pose 2", "Pose 3"])
        self.assertEqual(len(result.steps), 3)
        self.assertEqual(result.handedness, "Right")
        self.assertEqual(len(result.hold_ms), 3)
        self.assertEqual(len(result.gap_ms), 2)
        for measured, expected in zip(result.hold_ms, (400, 350, 450)):
            self.assertGreater(measured, expected * 0.45)
            self.assertLess(measured, expected * 1.6)
        self.assertGreaterEqual(result.max_gap_ms, max(result.gap_ms))
        self.assertLessEqual(result.dwell_ms, min(result.hold_ms))
        self.assertGreaterEqual(result.max_hold_ms, max(result.hold_ms))
        for step in result.steps:
            self.assertGreaterEqual(len(step.samples), 2)

    def test_too_few_poses_errors(self) -> None:
        from hgr.custom_gestures.pose_sequence_analysis import analyze_pose_sequence

        p = _synthetic_hand(mode="open")
        frames = _hold_pose_frames(p, t0=0.0, hold_s=0.8)
        with self.assertRaises(ValueError):
            analyze_pose_sequence(frames, expected_steps=3)


if __name__ == "__main__":
    unittest.main()
