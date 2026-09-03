"""Diagnostic: dynamic custom-gesture false fires and misses, offline.

Synthetic takes are streamed through the real classifier so the two
reported problems can be measured without a camera:

  entry   why bringing a hand into frame confirmed a "wave up"
  circle  why a circle gesture never fired, and how much speed
          variation each gesture actually tolerates

Run:  python tools/diag_dynamic_intent.py [scenario]
      scenario is one of: entry circle crossfire channels speed
                          threshold takespeed  (default: all)
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, "src")
sys.path.insert(0, "tests")

from hgr.custom_gestures.dynamic_classifier import (  # noqa: E402
    DynamicGestureClassifier,
    SpringMatcher,
    build_template_from_takes,
)
from hgr.custom_gestures.dynamic_recording import (  # noqa: E402
    DynamicGestureTake,
    build_dynamic_features,
)
from hgr.custom_gestures.key_point_selector import select_key_points  # noqa: E402
from test_dynamic_gesture_key_points import (  # noqa: E402
    NUM_LANDMARKS,
    _base_hand_landmarks,
    _curl_fingers_except,
)

OPEN = (4, 8, 12, 16, 20)


# ---------------------------------------------------------------------------
# synthetic takes


def _take(path_xy, *, duration, keep_open=OPEN):
    """Build a take whose wrist follows `path_xy` (palm units)."""
    frames = len(path_xy)
    pose = _curl_fingers_except(_base_hand_landmarks(), keep_open=keep_open)
    landmarks = np.empty((frames, NUM_LANDMARKS, 3), dtype=np.float32)
    wrists = np.empty((frames, 3), dtype=np.float32)
    timestamps = np.linspace(0.0, duration, frames, dtype=np.float64)
    for f in range(frames):
        frame = pose.copy()
        frame[:, 0] += path_xy[f][0]
        frame[:, 1] += path_xy[f][1]
        landmarks[f] = frame
        wrists[f] = frame[0].copy()
    return DynamicGestureTake(
        timestamps=timestamps,
        landmarks=landmarks,
        handedness="Right",
        raw_duration_seconds=float(duration),
        wrist_palm_scaled=wrists,
    )


def _straight_up(frames, total=1.4):
    return [(0.0, -total * f / max(1, frames - 1)) for f in range(frames)]


def _straight_right(frames, total=1.2):
    return [(total * f / max(1, frames - 1), 0.0) for f in range(frames)]


def _circle(frames, radius=1.0):
    """Start at 6 o'clock, travel left -> up -> right -> down."""
    out = []
    for f in range(frames):
        a = 2.0 * np.pi * f / max(1, frames - 1)
        out.append((-radius * np.sin(a), -radius * (np.cos(a) - 1.0)))
    return out


def _template(name, takes):
    return build_template_from_takes(name, takes, select_key_points(takes).indices)


def _feats(take, key_indices):
    rel = np.stack([f - f[0:1] for f in take.landmarks]).astype(np.float32)
    return build_dynamic_features(
        rel, take.wrist_palm_scaled, np.asarray(take.timestamps), key_indices,
    )


def _spring_min(template_feats, stream_feats):
    matcher = SpringMatcher(np.asarray(template_feats, dtype=np.float32))
    best = float("inf")
    for row in stream_feats:
        _c, _s, _r, mn, _ms, _mt = matcher.step(row)
        best = min(best, mn)
    return best


def _stream(classifier, take, *, fps, lead_still=0, t0=0.0):
    """Feed a take through the classifier frame by frame."""
    dt = 1.0 / fps
    t = t0
    fired = []
    first = take.landmarks[0]
    for _ in range(lead_still):
        rel = (first - first[0:1]).astype(np.float32)
        m = classifier.update(rel, t, wrist_palm_scaled=take.wrist_palm_scaled[0])
        if m is not None:
            fired.append((t, m))
        t += dt
    for f in range(take.num_frames):
        raw = take.landmarks[f]
        rel = (raw - raw[0:1]).astype(np.float32)
        m = classifier.update(rel, t, wrist_palm_scaled=take.wrist_palm_scaled[f])
        if m is not None:
            fired.append((t, m))
        t += dt
    return fired


# ---------------------------------------------------------------------------
# scenarios


def scenario_entry() -> None:
    print("=" * 70)
    print("entry) hand brought into frame vs a recorded 1.2 s wave up")
    takes = [_take(_straight_up(36), duration=1.2) for _ in range(3)]
    tpl = _template("wave_up", takes)
    print(f"   template: magnitude={tpl.intent_magnitude:.2f} palm units, "
          f"window={tpl.intent_window_seconds:.2f}s")

    for frac in (0.4, 0.6, 0.8, 1.0):
        clf = DynamicGestureClassifier([tpl])
        entry = _take(_straight_up(10, total=1.4 * frac), duration=0.30)
        fired = _stream(clf, entry, fps=33.0)
        print(f"   quick entry   travel={1.4 * frac:.2f} in 0.30 s "
              f"({frac:>4.0%} of recorded distance) -> "
              f"{'FIRED ' + fired[0][1].gesture_name if fired else 'no fire'}")

    for frac in (0.8, 1.0):
        clf = DynamicGestureClassifier([tpl])
        rise = _take(_straight_up(10, total=1.4 * frac), duration=0.30)
        fired = _stream(clf, rise, fps=33.0)
        hold = _take([(0.0, -1.4 * frac)] * 50, duration=1.5)
        fired += _stream(clf, hold, fps=33.0, t0=0.30)
        print(f"   raise + hold  travel={1.4 * frac:.2f} in 0.30 s then still "
              f"({frac:>4.0%} of recorded distance) -> "
              f"{'FIRED ' + fired[0][1].gesture_name if fired else 'no fire'}")

    clf = DynamicGestureClassifier([tpl])
    lead = _take([(0.0, 0.0)] * 20, duration=0.66)
    _stream(clf, lead, fps=30.0)
    real = _take(_straight_up(36), duration=1.2)
    fired = _stream(clf, real, fps=30.0, t0=0.66)
    print(f"   real full wave                                        -> "
          f"{'FIRED ' + fired[0][1].gesture_name if fired else 'NO FIRE (regression!)'}")


def scenario_circle() -> None:
    print("=" * 70)
    print("circle) a circle performed at different speeds, end to end")
    takes = [_take(_circle(45), duration=1.5) for _ in range(3)]
    tpl = _template("circle", takes)
    thr = float(DynamicGestureClassifier([tpl])._effective_thresholds[0])
    print(f"   stored threshold={tpl.match_threshold:.3f} effective={thr:.3f}   "
          f"intent magnitude={tpl.intent_magnitude:.4f} "
          f"(a circle has no net travel, so SPRING is the only path)")

    for label, frames, duration in (
        ("as recorded", 45, 1.5),
        ("30% slower", 59, 1.95),
        ("50% slower", 68, 2.25),
        ("2x slower", 90, 3.0),
    ):
        clf = DynamicGestureClassifier([tpl])
        live = _take(_circle(frames), duration=duration)
        fps = frames / duration
        fired = _stream(clf, live, fps=fps, lead_still=6)
        tail = _take([tuple(live.wrist_palm_scaled[-1][:2])] * 12, duration=0.4)
        fired += _stream(clf, tail, fps=fps, t0=duration + 6 / fps)
        cost = _spring_min(tpl.sample_features[0], _feats(live, list(tpl.key_point_indices)))
        print(f"   {label:<14} cost={cost:6.3f} vs {thr:.3f} -> "
              f"{'FIRED' if fired else 'no fire'}")


def scenario_crossfire() -> None:
    print("=" * 70)
    print("crossfire) circling with only wave-up registered")
    takes = [_take(_straight_up(36), duration=1.2) for _ in range(3)]
    tpl = _template("wave_up", takes)
    clf = DynamicGestureClassifier([tpl])
    live = _take(_circle(68), duration=2.25)
    fired = _stream(clf, live, fps=68 / 2.25, lead_still=6)
    names = [m.gesture_name for _t, m in fired]
    print(f"   fires: {names or 'none'}")


def scenario_channels() -> None:
    print("=" * 70)
    print("channels) which feature block carries the cost, and why")
    print("          direction-only velocity was NOT adopted")
    takes = [_take(_circle(45), duration=1.5) for _ in range(3)]
    tpl = _template("circle", takes)
    kp = list(tpl.key_point_indices)
    pos_end = 3 * len(kp)
    tf = tpl.sample_features[0]

    def unit_vel(feats):
        out = np.asarray(feats, dtype=np.float32).copy()
        vel = out[:, pos_end:]
        out[:, pos_end:] = vel / np.maximum(
            np.linalg.norm(vel, axis=1, keepdims=True), 1e-3,
        )
        return out

    print(f"   {'live':<16}{'full':>8}{'position':>10}{'velocity':>10}{'unit-vel':>10}")
    for label, frames, duration in (
        ("as recorded", 45, 1.5),
        ("50% slower", 68, 2.25),
        ("2x slower", 90, 3.0),
    ):
        lf = _feats(_take(_circle(frames), duration=duration), kp)
        print(f"   {label:<16}"
              f"{_spring_min(tf, lf):>8.3f}"
              f"{_spring_min(tf[:, :pos_end], lf[:, :pos_end]):>10.3f}"
              f"{_spring_min(tf[:, pos_end:], lf[:, pos_end:]):>10.3f}"
              f"{_spring_min(unit_vel(tf), unit_vel(lf)):>10.3f}")
    print("   -> position contributes nothing (a rigid hand's wrist-relative")
    print("      landmarks do not change), so speed-dependent velocity is the")
    print("      whole signal. Unit velocity fixes the speed sensitivity but")
    print("      then a wave-up template matches a circle's upward arc:")
    up_takes = [_take(_straight_up(36), duration=1.2) for _ in range(3)]
    up = _template("wave_up", up_takes)
    ukp = list(up.key_point_indices)
    upos = 3 * len(ukp)
    uf = up.sample_features[0]

    def unit_vel_up(feats):
        out = np.asarray(feats, dtype=np.float32).copy()
        vel = out[:, upos:]
        out[:, upos:] = vel / np.maximum(
            np.linalg.norm(vel, axis=1, keepdims=True), 1e-3,
        )
        return out

    circle_live = _feats(_take(_circle(45), duration=1.5), ukp)
    print(f"      wave-up template vs a circle: current={_spring_min(uf, circle_live):.3f} "
          f"unit-vel={_spring_min(unit_vel_up(uf), unit_vel_up(circle_live)):.3f}")


def scenario_speed() -> None:
    print("=" * 70)
    print("speed) how much speed variation each gesture tolerates")
    for gesture, path_fn, frames, dur in (
        ("circle", _circle, 45, 1.5),
        ("wave_up", _straight_up, 36, 1.2),
        ("swipe_right", _straight_right, 24, 0.8),
    ):
        takes = [_take(path_fn(frames), duration=dur) for _ in range(3)]
        tpl = _template(gesture, takes)
        kp = list(tpl.key_point_indices)
        thr = float(DynamicGestureClassifier([tpl])._effective_thresholds[0])
        print(f"\n   [{gesture}] recorded in {dur:.2f}s, stored threshold="
              f"{tpl.match_threshold:.3f}, effective={thr:.3f}")
        for ratio in (0.7, 0.85, 1.0, 1.15, 1.3, 1.5, 2.0):
            fr = max(6, int(round(frames * ratio)))
            live = _take(path_fn(fr), duration=dur * ratio)
            cost = _spring_min(tpl.sample_features[0], _feats(live, kp))
            print(f"     performed in {dur * ratio:4.2f}s ({ratio:>4.0%}) "
                  f"cost={cost:6.3f}  {'match' if cost < thr else 'MISS'}")


def scenario_threshold() -> None:
    print("=" * 70)
    print("threshold) motion-scaled floor: tolerance gained vs impostor margin")
    gestures = {
        "circle": (_circle, 45, 1.5),
        "wave_up": (_straight_up, 36, 1.2),
        "swipe_right": (_straight_right, 24, 0.8),
    }
    built = {
        name: _template(name, [_take(fn(fr), duration=du) for _ in range(3)])
        for name, (fn, fr, du) in gestures.items()
    }
    for name, (path_fn, frames, dur) in gestures.items():
        tpl = built[name]
        kp = list(tpl.key_point_indices)
        thr = float(DynamicGestureClassifier([tpl])._effective_thresholds[0])
        widest = None
        for ratio in (1.0, 1.15, 1.3, 1.5, 1.75, 2.0):
            fr = max(6, int(round(frames * ratio)))
            live = _take(path_fn(fr), duration=dur * ratio)
            if _spring_min(tpl.sample_features[0], _feats(live, kp)) < thr:
                widest = ratio
            else:
                break
        worst = float("inf")
        for other, (opath, ofr, odur) in gestures.items():
            if other == name:
                continue
            for r in (1.0, 1.5):
                live = _take(opath(int(ofr * r)), duration=odur * r)
                worst = min(worst, _spring_min(tpl.sample_features[0], _feats(live, kp)))
        still = _take([(0.0, 0.0)] * 40, duration=1.3)
        worst = min(worst, _spring_min(tpl.sample_features[0], _feats(still, kp)))
        print(f"   {name:<12} threshold {float(tpl.match_threshold):.2f} -> {thr:5.2f}  "
              f"tolerates up to {'n/a' if widest is None else f'{widest:.0%}'} duration  "
              f"worst impostor={worst:5.2f} "
              f"({'safe' if worst > thr else 'FALSE-POSITIVE RISK'})")


def scenario_takespeed() -> None:
    print("=" * 70)
    print("takespeed) recording the 3 takes at varied speeds widens the band")
    for label, ratios in (
        ("all 3 takes same speed", (1.0, 1.0, 1.0)),
        ("takes at 0.8x/1.0x/1.3x", (0.8, 1.0, 1.3)),
        ("takes at 0.7x/1.0x/1.5x", (0.7, 1.0, 1.5)),
    ):
        takes = [_take(_circle(int(45 * r)), duration=1.5 * r) for r in ratios]
        tpl = _template("circle", takes)
        thr = float(DynamicGestureClassifier([tpl])._effective_thresholds[0])
        kp = list(tpl.key_point_indices)
        band = []
        for ratio in (0.6, 0.7, 0.85, 1.0, 1.15, 1.3, 1.5, 1.75, 2.0):
            live = _take(_circle(int(45 * ratio)), duration=1.5 * ratio)
            lf = _feats(live, kp)
            if min(_spring_min(f, lf) for f in tpl.sample_features) < thr:
                band.append(ratio)
        span = f"{min(band):.0%}-{max(band):.0%}" if band else "none"
        print(f"   {label:<26} threshold={thr:.2f}  matches at {span} "
              f"of the recorded duration")


SCENARIOS = {
    "entry": scenario_entry,
    "circle": scenario_circle,
    "crossfire": scenario_crossfire,
    "channels": scenario_channels,
    "speed": scenario_speed,
    "threshold": scenario_threshold,
    "takespeed": scenario_takespeed,
}


if __name__ == "__main__":
    if os.environ.get("HGR_DYNAMIC_DEBUG") == "1":
        import logging
        logging.basicConfig(level=logging.WARNING)
    wanted = sys.argv[1:] or list(SCENARIOS)
    for key in wanted:
        fn = SCENARIOS.get(key)
        if fn is None:
            print(f"unknown scenario {key!r}; pick from {', '.join(SCENARIOS)}")
            raise SystemExit(2)
        fn()

# Author: Konstantin Markov
