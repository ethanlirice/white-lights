"""Per-frame squat-depth judgment — STUB.

The rule (IPF / USAPL, functionally identical): a squat reaches legal depth when
the **hip crease drops below the top of the knee**. This module answers, for a
single fused 3D frame, "is the subject below parallel right now, and how sure am
I?". It does *not* segment reps or emit a final verdict — that is `reps.py`.

Contract
--------
Input:  one :class:`~whitelights.types.FrameKeypoints3D` (fused 3D, +z up).
Output: a :class:`DepthFrameResult` for that frame.

Requirements the implementation must honour:
  * Depth metric: derive a hip-crease height and a top-of-knee height from the
    hip/knee keypoints (see ``DEPTH_KEYPOINTS``), then
    ``depth_margin = knee_top_z - hip_crease_z``. Positive == legal depth (hip
    below knee). Note COCO gives a hip *joint* centre, not the anatomical
    crease; account for the offset (TODO(ethan): calibrate).
  * Confidence gating: if the contributing keypoints' confidence is below
    ``config.min_confidence``, set ``gated=True`` and leave ``is_below_parallel``
    as ``None`` (unknown) rather than guessing — this is what lets `reps.py`
    return UNCERTAIN instead of a wrong call.
  * Bilateral handling: combine left/right sides sensibly (the rules judge the
    higher hip / the side facing the referee); TODO(ethan): decide policy.

TODO(ethan): implement `judge_depth_frame`. `judge_depth_sequence` is a thin
map provided for convenience once the per-frame logic exists.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from .types import FrameKeypoints3D, Pose3DSequence


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

    See the module contract. Must confidence-gate rather than guess.
    """
    raise NotImplementedError("TODO(ethan): per-frame depth judgment not implemented")


def judge_depth_sequence(
    sequence: Pose3DSequence, config: DepthConfig | None = None
) -> list[DepthFrameResult]:
    """Apply `judge_depth_frame` across a whole 3D sequence.

    Provided as glue; depends on `judge_depth_frame`, so it raises until that is
    implemented.
    """
    return [judge_depth_frame(f, config) for f in sequence.frames]
