"""Real-time signal filters for the live path.

The batch pipeline smooths a whole clip at once; live judging needs a *causal*
filter that runs frame-by-frame with low latency. The One-Euro filter is the
standard choice for interactive pose: it smooths hard when the signal is still
(killing jitter) and backs off when the signal moves fast (preserving
responsiveness), via a velocity-dependent cutoff.

Reference: Casiez, Roussel & Vogel, "1€ Filter" (CHI 2012).
"""

from __future__ import annotations

import math

from .types import FrameKeypoints, Keypoint2D


class OneEuroFilter:
    """Scalar One-Euro filter. Call with a monotonically increasing timestamp."""

    def __init__(
        self, *, min_cutoff: float = 1.0, beta: float = 0.0, d_cutoff: float = 1.0
    ) -> None:
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self._x: float | None = None
        self._dx = 0.0
        self._t: float | None = None

    @staticmethod
    def _alpha(cutoff: float, dt: float) -> float:
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def reset(self) -> None:
        self._x = None
        self._dx = 0.0
        self._t = None

    def __call__(self, t: float, x: float) -> float:
        if self._t is None or self._x is None:
            self._t, self._x, self._dx = t, x, 0.0
            return x
        dt = t - self._t
        if dt <= 0:
            dt = 1e-3
        dx = (x - self._x) / dt
        a_d = self._alpha(self.d_cutoff, dt)
        dx_hat = a_d * dx + (1.0 - a_d) * self._dx
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = self._alpha(cutoff, dt)
        x_hat = a * x + (1.0 - a) * self._x
        self._t, self._x, self._dx = t, x_hat, dx_hat
        return x_hat


class StreamingKeypointSmoother:
    """One-Euro filters per keypoint coordinate, with confidence gating.

    Low-confidence keypoints are dropped (not fed to the filters), so a garbage
    detection neither pollutes the smoothed track nor advances the filter state.
    """

    def __init__(
        self, *, min_cutoff: float = 1.0, beta: float = 0.02, min_confidence: float = 0.3
    ) -> None:
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.min_confidence = min_confidence
        self._fx: dict[str, OneEuroFilter] = {}
        self._fy: dict[str, OneEuroFilter] = {}

    def _filter(self, store: dict[str, OneEuroFilter], name: str) -> OneEuroFilter:
        if name not in store:
            store[name] = OneEuroFilter(min_cutoff=self.min_cutoff, beta=self.beta)
        return store[name]

    def smooth(self, frame: FrameKeypoints) -> FrameKeypoints:
        out: dict[str, Keypoint2D] = {}
        for name, kp in frame.keypoints.items():
            if kp.confidence < self.min_confidence:
                continue
            x = self._filter(self._fx, name)(frame.time_s, kp.x)
            y = self._filter(self._fy, name)(frame.time_s, kp.y)
            out[name] = Keypoint2D(name=name, x=x, y=y, confidence=kp.confidence)
        return FrameKeypoints(
            frame_idx=frame.frame_idx,
            time_s=frame.time_s,
            keypoints=out,
            detected=bool(out),
            subject_confidence=frame.subject_confidence,
        )
