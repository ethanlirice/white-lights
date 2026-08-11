"""Per-frame squat-depth judgment.

The rule (IPF / USAPL, functionally identical): a squat reaches legal depth when
the **hip crease drops below the top of the knee**. This module answers, for a
single fused 3D frame, "is the subject below parallel right now, and how sure am
I?". It does *not* segment reps or emit a final verdict — that is `reps.py`.

Contract
--------
Input:  one :class:`~whitelights.types.FrameKeypoints3D` (fused 3D, +z up).
Output: a :class:`DepthFrameResult` for that frame.

Implementation decisions
------------------------
  * Depth metric: ``depth_margin = knee_top_z - hip_crease_z`` in world units.
    Positive == legal depth (hip crease below top of knee). The below-parallel
    *call* is the pure sign of this margin, so it is scale-invariant — it holds
    whether ``z`` is in pixels (single-camera fallback) or metric units (real
    triangulation). Only absolute magnitudes differ between those modes.
  * Hip crease: COCO gives a hip *joint* centre, so we drop it by
    ``config.hip_crease_offset`` (world units, default 0 == uncalibrated) to
    approximate the anatomical crease.
  * Bilateral policy: judge the **higher** hip (``max z``) against the **higher**
    knee (``max z``). Judging the higher/shallower hip is the conservative call —
    if it has broken parallel, the lower one certainly has.
  * Confidence gating: aggregate confidence is the **minimum** over the
    contributing hip/knee keypoints (weakest link). Below ``min_confidence`` — or
    a hip/knee entirely missing — yields ``gated=True`` and
    ``is_below_parallel=None`` (unknown) rather than a guess, which is what lets
    `reps.py` return UNCERTAIN.

``judge_depth_sequence`` is a thin map over ``judge_depth_frame``.
"""

from __future__ import annotations

import math

from pydantic import BaseModel, Field

from .types import FrameKeypoints3D, Keypoint3D, Pose3DSequence

_HIP_KEYPOINTS = ("left_hip", "right_hip")
_KNEE_KEYPOINTS = ("left_knee", "right_knee")


class DepthConfig(BaseModel):
    """Tunables for the depth judge."""

    min_confidence: float = Field(default=0.4, ge=0.0, le=1.0)
    # Vertical offset from the COCO hip-joint keypoint down to the anatomical hip
    # crease, in **absolute world units**. Only meaningful once fusion produces
    # metric 3D; in the single-camera fallback `z` is pixels, so a metric value
    # here is off by orders of magnitude. Prefer the fraction below.
    hip_crease_offset: float = 0.0
    # The same offset expressed as a fraction of the lifter's thigh length,
    # measured per frame. Scale-invariant, so it means the same thing in pixel
    # space and in metric space — matching how every other threshold in this
    # codebase is expressed. Anatomically the crease sits roughly 0.14 thigh
    # below the joint centre; left at 0.0 until validated against labelled clips.
    hip_crease_thigh_fraction: float = 0.0


class DepthFrameResult(BaseModel):
    """The depth assessment for a single frame."""

    frame_idx: int
    time_s: float
    # None when confidence-gated (unknown), else True/False.
    is_below_parallel: bool | None = None
    # Signed; positive == hip crease below top of knee. None when gated.
    depth_margin: float | None = None
    confidence: float = 0.0
    gated: bool = False
    # The two heights the call actually compares (world z, +z up). Surfaced so
    # the overlay can *draw* the comparison instead of re-deriving it: which hip,
    # which knee, and how far the crease offset drops the landmark are decisions
    # made here, and a second implementation in JavaScript would drift from them.
    hip_crease_z: float | None = None
    knee_top_z: float | None = None


def judge_depth_frame(
    frame: FrameKeypoints3D, config: DepthConfig | None = None
) -> DepthFrameResult:
    """Judge whether the subject is below parallel in a single frame.

    See the module contract. Confidence-gates rather than guessing.
    """
    config = config or DepthConfig()

    hips: list[Keypoint3D] = [kp for name in _HIP_KEYPOINTS if (kp := frame.get(name)) is not None]
    knees: list[Keypoint3D] = [
        kp for name in _KNEE_KEYPOINTS if (kp := frame.get(name)) is not None
    ]

    # Can't judge without at least one hip and one knee.
    if not hips or not knees:
        return _gated(frame, confidence=0.0)

    confidence = min(kp.confidence for kp in (*hips, *knees))
    if confidence < config.min_confidence:
        return _gated(frame, confidence=confidence)

    # Higher (shallower) hip crease vs. higher knee — the conservative pairing.
    offset = config.hip_crease_offset + config.hip_crease_thigh_fraction * _thigh_length(frame)
    hip_crease_z = max(kp.z - offset for kp in hips)
    knee_top_z = max(kp.z for kp in knees)
    margin = knee_top_z - hip_crease_z

    return DepthFrameResult(
        frame_idx=frame.frame_idx,
        time_s=frame.time_s,
        is_below_parallel=margin > 0,
        depth_margin=margin,
        confidence=confidence,
        gated=False,
        hip_crease_z=hip_crease_z,
        knee_top_z=knee_top_z,
    )


def _thigh_length(frame: FrameKeypoints3D) -> float:
    """Mean hip->knee distance, the body reference the crease fraction scales.

    Returns 0.0 when unmeasurable, which makes the fractional offset a no-op
    rather than a guess.
    """
    lengths: list[float] = []
    for side in ("left", "right"):
        hip, knee = frame.get(f"{side}_hip"), frame.get(f"{side}_knee")
        if hip is None or knee is None:
            continue
        lengths.append(math.dist((hip.x, hip.y, hip.z), (knee.x, knee.y, knee.z)))
    return sum(lengths) / len(lengths) if lengths else 0.0


def _gated(frame: FrameKeypoints3D, *, confidence: float) -> DepthFrameResult:
    """A frame we decline to judge (missing or low-confidence keypoints)."""
    return DepthFrameResult(
        frame_idx=frame.frame_idx,
        time_s=frame.time_s,
        is_below_parallel=None,
        depth_margin=None,
        confidence=confidence,
        gated=True,
    )


def judge_depth_sequence(
    sequence: Pose3DSequence, config: DepthConfig | None = None
) -> list[DepthFrameResult]:
    """Apply `judge_depth_frame` across a whole 3D sequence.

    Provided as glue; depends on `judge_depth_frame`, so it raises until that is
    implemented.
    """
    return [judge_depth_frame(f, config) for f in sequence.frames]
