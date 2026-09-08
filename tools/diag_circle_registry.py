"""Inspect saved circle gesture + rebuild features at varied speeds."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from hgr.custom_gestures.dynamic_classifier import SpringMatcher
from hgr.custom_gestures.dynamic_recording import build_dynamic_features
from hgr.custom_gestures.dynamic_runtime import DynamicGestureRuntime


def _rebuild_feats(sample, wrist, key_indices, *, duration_s: float):
    sample = np.asarray(sample, dtype=np.float32)
    wrist = np.asarray(wrist, dtype=np.float32)
    T = sample.shape[0]
    # sample is wrist-relative already; rebuild full 21 landmark cloud
    lm = np.zeros((T, 21, 3), dtype=np.float32)
    for col, idx in enumerate(key_indices):
        if 0 <= idx < 21 and col < sample.shape[1]:
            lm[:, idx, :] = sample[:, col, :]
    ts = np.linspace(0.0, duration_s, T, dtype=np.float64)
    return build_dynamic_features(lm, wrist, ts, list(key_indices))


def main() -> None:
    p = Path.home() / ".hgr_app" / "custom_gestures.json"
    raw = json.loads(p.read_text(encoding="utf-8"))
    gestures = raw.get("gestures") or []
    rt = DynamicGestureRuntime()
    rt.reload()
    clf = rt._classifier
    assert clf is not None

    circle = next(g for g in gestures if g.get("name") == "circle")
    kp = [int(i) for i in circle["key_point_indices"]]
    sf = circle["sample_features"]
    wt = circle["wrist_trajectories"]
    st = circle["sample_trajectories"]
    tpl_idx = next(i for i, t in enumerate(clf._templates) if t.name == "circle")
    thr = float(clf._effective_thresholds[tpl_idx])
    print(f"circle thr={thr:.3f} stored_thr={circle.get('match_threshold')} kp={kp}")

    # Recorded duration unknown after resample — estimate from velocity.
    # Template features were built with real take timestamps then resampled.
    # Probe: rebuild at 1.5s / 2.0s / 2.5s / 3.0s and compare to stored feat0.
    feat0 = np.asarray(sf[0], dtype=np.float32)
    print("rebuild duration that best matches stored features:")
    for dur in (1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0):
        rebuilt = _rebuild_feats(st[0], wt[0], kp, duration_s=dur)
        # Compare to stored (ignore row0 velocity weirdness)
        err = float(np.mean(np.abs(rebuilt[1:] - feat0[1:])))
        print(f"  dur={dur:.2f}s mean_abs_err={err:.4f}")

    # Pick best duration
    best_dur, best_err = min(
        (
            (dur, float(np.mean(np.abs(
                _rebuild_feats(st[0], wt[0], kp, duration_s=dur)[1:] - feat0[1:]
            ))))
            for dur in (1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0)
        ),
        key=lambda x: x[1],
    )
    print(f"best recorded-duration estimate: {best_dur:.2f}s (err={best_err:.4f})")

    print("\nlive speed tolerance (rebuild features with real dt):")
    for ratio in (0.7, 0.85, 1.0, 1.15, 1.3, 1.5, 2.0):
        live = _rebuild_feats(st[0], wt[0], kp, duration_s=best_dur * ratio)
        best = float("inf")
        for tpl_feat in sf:
            m = SpringMatcher(np.asarray(tpl_feat, dtype=np.float32))
            for row in live:
                _c, _s, _r, mn, _ms, _mt = m.step(row)
                best = min(best, mn)
        print(
            f"  {ratio:>4.0%} duration  cost={best:6.3f} vs {thr:.3f}  "
            f"{'match' if best < thr else 'MISS'}"
        )

    # Cross costs with real rebuild at best_dur
    live = _rebuild_feats(st[0], wt[0], kp, duration_s=best_dur)
    print("\ncross-template costs for a circle stream:")
    for i, tpl in enumerate(clf._templates):
        if not tpl.sample_features:
            continue
        if tpl.sample_features[0].shape[1] != live.shape[1]:
            print(f"  {tpl.name}: dim mismatch")
            continue
        m = SpringMatcher(np.asarray(tpl.sample_features[0], dtype=np.float32))
        best = float("inf")
        for row in live:
            _c, _s, _r, mn, _ms, _mt = m.step(row)
            best = min(best, mn)
        print(f"  {tpl.name}: {best:.3f} / {clf._effective_thresholds[i]:.3f}")


if __name__ == "__main__":
    main()

# Author: Konstantin Markov
