"""Postural / foot faults derived from body geometry (v2.3).

These faults are evaluated from the fused 3D keypoints, independent of the depth
judgment. `reps.py` calls them at the relevant moments of a rep and folds any
result into the rep's fault list.

Implemented (tractable from COCO-17 keypoints):
  * INCOMPLETE_LOCKOUT — at lockout (rep start / finish) the knees must be
    locked, i.e. the hip-knee-ankle angle is near-straight. A knee angle below
    ``lockout_knee_angle_deg`` on either leg means not locked.
  * FOOT_MOVEMENT — an ankle that drifts horizontally during the rep by more
    than ``foot_movement_fraction`` of the lifter's thigh length (a body-scale
    reference, so the threshold is unit-invariant).

Deferred:
  * BAR_SUPPORTED_ON_THIGHS — not derivable from body keypoints alone; needs a
    bar / barbell detection signal. TODO(ethan).

Design note: every detector is *conservative* — it returns ``None`` ("can't
tell") when the keypoints it needs are missing or below confidence, so callers
never fault on absent signal. This is why a clip without ankle keypoints simply
yields no postural faults rather than false positives.
"""

from __future__ import annotations

import math

from pydantic import BaseModel, Field

from .types import FrameKeypoints3D, Keypoint3D

_SIDES = ("left", "right")


class PostureConfig(BaseModel):
    """Tunables for the postural detectors."""

    min_confidence: float = Field(default=0.4, ge=0.0, le=1.0)
    # Knee angle (deg) at or above which the leg counts as locked out.
    lockout_knee_angle_deg: float = 160.0
    # Elbow angle (deg) at or above which the arm counts as locked out (bench).
    lockout_elbow_angle_deg: float = 155.0
    # Ankle horizontal drift that trips FOOT_MOVEMENT, as a fraction of thigh
    # length (scale-invariant body reference).
    foot_movement_fraction: float = 0.15


def joint_angle_deg(
    frame: FrameKeypoints3D, a_name: str, b_name: str, c_name: str, min_confidence: float
) -> float | None:
    """Interior angle (degrees) at joint ``b`` for the chain a-b-c, or None.

    None when any keypoint is missing or below ``min_confidence``. The building
    block for both knee (hip-knee-ankle) and elbow (shoulder-elbow-wrist) angles.

    NOTE: a "locked" joint rarely reads as a clean 180 deg from a pose estimator
    (keypoint noise + anatomy vary), so callers should threshold well below 180
    and/or calibrate against the individual's own locked angle rather than an
    absolute — see the trackers' per-lifter lockout calibration.
    """
    a = frame.get(a_name)
    b = frame.get(b_name)
    c = frame.get(c_name)
    if a is None or b is None or c is None:
        return None
    if min(a.confidence, b.confidence, c.confidence) < min_confidence:
        return None
    return _angle_deg(_vec(b, a), _vec(b, c))


def knee_angle_deg(frame: FrameKeypoints3D, side: str, min_confidence: float) -> float | None:
    """Interior hip-knee-ankle angle (degrees) for one leg, or None if unknown."""
    return joint_angle_deg(frame, f"{side}_hip", f"{side}_knee", f"{side}_ankle", min_confidence)


def elbow_angle_deg(frame: FrameKeypoints3D, side: str, min_confidence: float) -> float | None:
    """Interior shoulder-elbow-wrist angle (degrees) for one arm, or None (bench)."""
    return joint_angle_deg(
        frame, f"{side}_shoulder", f"{side}_elbow", f"{side}_wrist", min_confidence
    )


def arms_locked(frame: FrameKeypoints3D, config: PostureConfig) -> bool | None:
    """Are both elbows locked in this frame? The more-bent arm governs.

    Returns True/False when at least one arm is measurable, else None.
    """
    angles = [
        a
        for side in _SIDES
        if (a := elbow_angle_deg(frame, side, config.min_confidence)) is not None
    ]
    if not angles:
        return None
    return min(angles) >= config.lockout_elbow_angle_deg


def is_locked_out(frame: FrameKeypoints3D, config: PostureConfig) -> bool | None:
    """Are both knees locked in this frame?

    Returns True/False when at least one leg is measurable (the more-bent leg
    governs), or None when neither leg can be measured.
    """
    angles = [
        a
        for side in _SIDES
        if (a := knee_angle_deg(frame, side, config.min_confidence)) is not None
    ]
    if not angles:
        return None
    return min(angles) >= config.lockout_knee_angle_deg


def foot_displacement_ratio(frames: list[FrameKeypoints3D], config: PostureConfig) -> float | None:
    """Max horizontal ankle drift over ``frames``, as a fraction of thigh length.

    Returns None when there is no usable ankle track or no thigh reference.
    """
    reference = _mean_thigh_length(frames, config.min_confidence)
    if not reference:
        return None

    max_ratio: float | None = None
    for side in _SIDES:
        points = [
            (a.x, a.y)
            for f in frames
            if (a := f.get(f"{side}_ankle")) is not None and a.confidence >= config.min_confidence
        ]
        if len(points) < 2:
            continue
        cx = sum(p[0] for p in points) / len(points)
        cy = sum(p[1] for p in points) / len(points)
        deviation = max(math.hypot(p[0] - cx, p[1] - cy) for p in points)
        ratio = deviation / reference
        max_ratio = ratio if max_ratio is None else max(max_ratio, ratio)
    return max_ratio


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _vec(a: Keypoint3D, b: Keypoint3D) -> tuple[float, float, float]:
    return (b.x - a.x, b.y - a.y, b.z - a.z)


def _angle_deg(u: tuple[float, float, float], v: tuple[float, float, float]) -> float | None:
    nu = math.sqrt(sum(c * c for c in u))
    nv = math.sqrt(sum(c * c for c in v))
    if nu == 0 or nv == 0:
        return None
    cos = sum(a * b for a, b in zip(u, v, strict=True)) / (nu * nv)
    return math.degrees(math.acos(max(-1.0, min(1.0, cos))))


def _mean_thigh_length(frames: list[FrameKeypoints3D], min_confidence: float) -> float | None:
    lengths: list[float] = []
    for f in frames:
        for side in _SIDES:
            hip = f.get(f"{side}_hip")
            knee = f.get(f"{side}_knee")
            if hip is None or knee is None:
                continue
            if min(hip.confidence, knee.confidence) < min_confidence:
                continue
            lengths.append(math.dist((hip.x, hip.y, hip.z), (knee.x, knee.y, knee.z)))
    if not lengths:
        return None
    return sum(lengths) / len(lengths)
