"""Keypoint time-series smoothing — STUB.

Contract
--------
Input:  a raw :class:`~whitelights.types.PoseSequence` straight from `pose.py`,
        where per-keypoint tracks are noisy and may contain gaps (frames where a
        keypoint was occluded or the subject undetected -> ``detected=False`` /
        low confidence).
Output: a `PoseSequence` of **identical length and time base** (same
        ``frame_idx`` / ``time_s`` per frame, same ``fps``, same ``camera_id``)
        whose keypoint tracks are de-jittered and gap-filled.

Requirements the implementation must honour:
  * Timing preserved: frame count and timestamps are unchanged; this is a
    filter over the series, never a resample.
  * Confidence-aware: low-confidence samples should be down-weighted or treated
    as gaps, not trusted equally with clean detections.
  * Causal-friendly: prefer a filter that can later run online (real-time),
    e.g. One-Euro / Kalman, rather than a whole-clip acausal fit — though the
    batch entry point here may look ahead.
  * Bounded interpolation: only bridge short gaps; a long dropout should remain
    a gap (leave the keypoint absent / confidence 0) rather than hallucinate.

TODO(ethan): implement. Candidate approaches: per-keypoint One-Euro filter, or
a constant-velocity Kalman filter with confidence as measurement noise.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from .types import PoseSequence


class SmoothingConfig(BaseModel):
    """Tunables for the smoother. Extend as the implementation demands."""

    min_confidence: float = Field(default=0.3, ge=0.0, le=1.0)
    max_gap_frames: int = Field(default=5, ge=0)
    # One-Euro-style params (used only if that filter is chosen).
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
    raise NotImplementedError("TODO(ethan): smoothing not implemented")
