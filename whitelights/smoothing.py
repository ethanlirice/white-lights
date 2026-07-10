"""Keypoint time-series smoothing.

Contract
--------
Input:  a raw :class:`~whitelights.types.PoseSequence` straight from `pose.py`,
        where per-keypoint tracks are noisy and may contain gaps (frames where a
        keypoint was occluded or the subject undetected -> ``detected=False`` /
        low confidence).
Output: a `PoseSequence` of **identical length and time base** (same
        ``frame_idx`` / ``time_s`` per frame, same ``fps``, same ``camera_id``)
        whose keypoint tracks are gap-filled and confidence-gated.

What this implementation does (v2.0):
  * Timing preserved: frame count and timestamps are copied through unchanged;
    this is a filter over the series, never a resample.
  * Confidence-aware: a sample below ``config.min_confidence`` is treated as a
    gap, not trusted equally with a clean detection.
  * Bounded interpolation: an *interior* gap of at most ``max_gap_frames`` is
    bridged by linear interpolation between its anchors (interpolated points get
    the lower anchor confidence, flagging them as inferred). Longer gaps, and
    leading/trailing gaps with no anchor on one side, stay absent.

Deferred (TODO(ethan)): genuine jitter reduction. Clean samples are currently
passed through unchanged — the ``min_cutoff`` / ``beta`` config fields are
reserved for a causal One-Euro (or constant-velocity Kalman) filter that would
replace the passthrough without changing this signature.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from .types import FrameKeypoints, Keypoint2D, PoseSequence

# A per-frame keypoint sample: (x, y, confidence), or None when absent/gated.
_Sample = tuple[float, float, float] | None


class SmoothingConfig(BaseModel):
    """Tunables for the smoother. Extend as the implementation demands."""

    min_confidence: float = Field(default=0.3, ge=0.0, le=1.0)
    max_gap_frames: int = Field(default=5, ge=0)
    # One-Euro-style params (reserved for the deferred jitter filter).
    min_cutoff: float = 1.0
    beta: float = 0.0


def smooth_sequence(sequence: PoseSequence, config: SmoothingConfig | None = None) -> PoseSequence:
    """Smooth and gap-fill a single-camera pose track. See module contract.

    Args:
        sequence: raw per-frame keypoints for one camera.
        config: smoothing tunables; defaults applied when ``None``.

    Returns:
        A new `PoseSequence` with the same time base and cleaned keypoint tracks.
    """
    config = config or SmoothingConfig()
    frames = sequence.frames
    n = len(frames)

    names: set[str] = set()
    for f in frames:
        names.update(f.keypoints.keys())

    # Rebuild each keypoint track independently, then reassemble per frame.
    rebuilt: list[dict[str, Keypoint2D]] = [{} for _ in range(n)]
    for name in names:
        series = _extract_series(frames, name, config.min_confidence)
        series = _interpolate_gaps(series, config.max_gap_frames)
        for i, sample in enumerate(series):
            if sample is not None:
                x, y, conf = sample
                rebuilt[i][name] = Keypoint2D(name=name, x=x, y=y, confidence=conf)

    out_frames: list[FrameKeypoints] = []
    for i, f in enumerate(frames):
        keypoints = rebuilt[i]
        confs = [kp.confidence for kp in keypoints.values()]
        out_frames.append(
            FrameKeypoints(
                frame_idx=f.frame_idx,
                time_s=f.time_s,
                keypoints=keypoints,
                detected=bool(keypoints),
                subject_confidence=sum(confs) / len(confs) if confs else 0.0,
            )
        )

    return PoseSequence(
        camera_id=sequence.camera_id,
        fps=sequence.fps,
        frames=out_frames,
        source=sequence.source,
    )


def _extract_series(
    frames: list[FrameKeypoints], name: str, min_confidence: float
) -> list[_Sample]:
    """Pull one keypoint's per-frame samples, gating sub-threshold ones to None."""
    series: list[_Sample] = []
    for f in frames:
        kp = f.get(name)
        if kp is None or kp.confidence < min_confidence:
            series.append(None)
        else:
            series.append((kp.x, kp.y, kp.confidence))
    return series


def _interpolate_gaps(series: list[_Sample], max_gap_frames: int) -> list[_Sample]:
    """Linearly bridge interior gaps up to ``max_gap_frames`` long.

    Leading/trailing gaps (no anchor on one side) and gaps longer than the limit
    are left as None.
    """
    out = list(series)
    n = len(out)
    i = 0
    while i < n:
        if out[i] is not None:
            i += 1
            continue
        # Gap runs over [i, j).
        j = i
        while j < n and out[j] is None:
            j += 1
        left, right = i - 1, j
        gap_len = j - i
        if left >= 0 and right < n and gap_len <= max_gap_frames:
            x0, y0, c0 = series[left]
            x1, y1, c1 = series[right]
            span = right - left
            for k in range(i, j):
                t = (k - left) / span
                out[k] = (x0 + (x1 - x0) * t, y0 + (y1 - y0) * t, min(c0, c1))
        i = j
    return out
