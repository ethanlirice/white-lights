"""Tests for the real-time One-Euro filter and streaming keypoint smoother."""

from __future__ import annotations

from whitelights.filters import OneEuroFilter, StreamingKeypointSmoother
from whitelights.types import FrameKeypoints, Keypoint2D


def test_first_sample_passes_through() -> None:
    f = OneEuroFilter(min_cutoff=1.0, beta=0.0)
    assert f(0.0, 5.0) == 5.0


def test_reduces_jitter_around_constant() -> None:
    # A noisy constant signal: the filter output should vary far less than the input.
    f = OneEuroFilter(min_cutoff=0.5, beta=0.0)
    noisy = [10.0, 12.0, 8.0, 11.0, 9.0, 13.0, 7.0, 10.5, 9.5, 11.5]
    out = [f(i / 30.0, x) for i, x in enumerate(noisy)]
    in_range = max(noisy) - min(noisy)
    out_range = max(out[2:]) - min(out[2:])  # skip warm-up
    assert out_range < in_range / 2


def test_tracks_a_ramp() -> None:
    # A steadily rising signal should be followed (small lag, not flattened).
    f = OneEuroFilter(min_cutoff=1.0, beta=0.1)
    out = [f(i / 30.0, float(i)) for i in range(30)]
    assert out[-1] > out[0]
    assert out[-1] > 20.0  # followed most of the way to 29


def test_smoother_drops_low_confidence_keypoints() -> None:
    smoother = StreamingKeypointSmoother(min_confidence=0.5)
    frame = FrameKeypoints(
        frame_idx=0,
        time_s=0.0,
        keypoints={
            "left_hip": Keypoint2D(name="left_hip", x=100.0, y=200.0, confidence=0.9),
            "right_hip": Keypoint2D(name="right_hip", x=110.0, y=205.0, confidence=0.1),
        },
        detected=True,
        subject_confidence=0.5,
    )
    out = smoother.smooth(frame)
    assert out.get("left_hip") is not None
    assert out.get("right_hip") is None  # gated out


def test_smoother_preserves_frame_metadata() -> None:
    smoother = StreamingKeypointSmoother()
    frame = FrameKeypoints(
        frame_idx=7,
        time_s=0.25,
        keypoints={"left_hip": Keypoint2D(name="left_hip", x=1.0, y=2.0, confidence=0.9)},
        detected=True,
        subject_confidence=0.9,
    )
    out = smoother.smooth(frame)
    assert out.frame_idx == 7
    assert out.time_s == 0.25
