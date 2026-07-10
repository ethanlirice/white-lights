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

from pydantic import BaseModel, Field

from .types import FrameKeypoints3D, Keypoint3D, Pose3DSequence

_HIP_KEYPOINTS = ("left_hip", "right_hip")
_KNEE_KEYPOINTS = ("left_knee", "right_knee")


class DepthConfig(BaseModel):
    """Tunables for the depth judge."""

    min_confidence: float = Field(default=0.4, ge=0.0, le=1.0)
    # Vertical offset (world units) from the COCO hip-joint keypoint down to the
    # anatomical hip crease. Calibrated, not guessed. TODO(ethan).
    hip_crease_offset: float = 0.0


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
    hip_crease_z = max(kp.z - config.hip_crease_offset for kp in hips)
    knee_top_z = max(kp.z for kp in knees)
    margin = knee_top_z - hip_crease_z

    return DepthFrameResult(
        frame_idx=frame.frame_idx,
        time_s=frame.time_s,
        is_below_parallel=margin > 0,
        depth_margin=margin,
        confidence=confidence,
        gated=False,
    )


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
