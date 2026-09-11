from __future__ import annotations

import math

import numpy as np


class OneEuroFilter:
    """1-D One Euro filter (Casiez et al. 2012) for interactive pointers.

    Trades jitter against lag adaptively: heavy smoothing when the input
    is slow (kills tremor while the user holds still) and low lag when
    the input is fast (so quick strokes don't rubber-band). Purpose-built
    for fingertip cursor input where a fixed EMA either feels fidgety at
    rest or laggy in motion.

    Tuning (inputs are normalized [0,1] coords, timestamps in seconds):
      - min_cutoff: lower => more smoothing at rest (less jitter, more lag)
      - beta:       higher => less lag during fast motion
    """

    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.5, d_cutoff: float = 1.0):
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self._x_prev: float | None = None
        self._dx_prev: float = 0.0
        self._t_prev: float | None = None

    def reset(self) -> None:
        self._x_prev = None
        self._dx_prev = 0.0
        self._t_prev = None

    @staticmethod
    def _alpha(cutoff: float, dt: float) -> float:
        tau = 1.0 / (2.0 * math.pi * max(cutoff, 1e-6))
        return 1.0 / (1.0 + tau / max(dt, 1e-6))

    def update(self, x: float, t: float, *, max_dt: float | None = None) -> float:
        x = float(x)
        t = float(t)
        if self._x_prev is None or self._t_prev is None:
            self._x_prev = x
            self._t_prev = t
            self._dx_prev = 0.0
            return x
        dt = t - self._t_prev
        if dt <= 0.0:
            dt = 1e-3
        # A dropped frame makes dt huge, so alpha jumps and landmark
        # jitter is treated as real motion (cursor crawl / drift).
        if max_dt is not None and dt > float(max_dt):
            dt = float(max_dt)
        dx = (x - self._x_prev) / dt
        a_d = self._alpha(self.d_cutoff, dt)
        dx_hat = a_d * dx + (1.0 - a_d) * self._dx_prev
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = self._alpha(cutoff, dt)
        x_hat = a * x + (1.0 - a) * self._x_prev
        self._x_prev = x_hat
        self._dx_prev = dx_hat
        self._t_prev = t
        return x_hat


class AdaptiveLandmarkSmoother:
    # r45: added `motion_normalizer` and `base_weight` tunables so an
    # opt-in aggressive Lite path can push toward a snappier fast-motion
    # response. Defaults preserve pre-r45 behaviour byte-for-byte
    # (0.040 / 0.48). All existing 3-arg callers keep the exact same
    # numerics — the new kwargs are additive.
    def __init__(
        self,
        alpha: float = 0.58,
        min_alpha: float = 0.18,
        max_alpha: float = 0.76,
        motion_normalizer: float = 0.040,
        base_weight: float = 0.48,
    ):
        self.alpha = float(alpha)
        self.min_alpha = float(min_alpha)
        self.max_alpha = float(max_alpha)
        # Clamp to safe ranges — zero normalizer would divide-by-zero;
        # a base_weight outside [0,1] would push adaptive_alpha outside
        # the [min,max] clamp intermittently.
        self._motion_normalizer = max(1e-3, float(motion_normalizer))
        self._base_weight = max(0.0, min(1.0, float(base_weight)))
        self._adaptive_weight = 1.0 - self._base_weight
        self._state: np.ndarray | None = None

    def reset(self) -> None:
        self._state = None

    def update(self, landmarks: np.ndarray) -> np.ndarray:
        current = landmarks.astype(np.float32)
        if self._state is None:
            self._state = current.copy()
            return self._state.copy()

        delta = current - self._state
        weighted = delta.copy()
        weighted[:, 2] *= 0.60
        motion = float(np.median(np.linalg.norm(weighted, axis=1)))
        adaptive_alpha = self.min_alpha + (self.max_alpha - self.min_alpha) * min(
            1.0, motion / self._motion_normalizer
        )
        adaptive_alpha = self._adaptive_weight * adaptive_alpha + self._base_weight * self.alpha
        adaptive_alpha = max(self.min_alpha, min(self.max_alpha, adaptive_alpha))
        self._state = adaptive_alpha * current + (1.0 - adaptive_alpha) * self._state
        return self._state.copy()

# Author: Konstantin Markov
