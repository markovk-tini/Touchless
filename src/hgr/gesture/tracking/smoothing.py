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

    def update(self, x: float, t: float) -> float:
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
    def __init__(self, alpha: float = 0.58, min_alpha: float = 0.18, max_alpha: float = 0.76):
        self.alpha = float(alpha)
        self.min_alpha = float(min_alpha)
        self.max_alpha = float(max_alpha)
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
        adaptive_alpha = self.min_alpha + (self.max_alpha - self.min_alpha) * min(1.0, motion / 0.040)
        adaptive_alpha = 0.52 * adaptive_alpha + 0.48 * self.alpha
        adaptive_alpha = max(self.min_alpha, min(self.max_alpha, adaptive_alpha))
        self._state = adaptive_alpha * current + (1.0 - adaptive_alpha) * self._state
        return self._state.copy()

# Author: Konstantin Markov
