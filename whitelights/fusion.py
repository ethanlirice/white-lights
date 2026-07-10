"""Multi-view fusion: 2D-per-camera -> 3D world coordinates — STUB.

This is the seam introduced by the multi-camera / 3D design decision. Each
camera yields a 2D :class:`~whitelights.types.PoseSequence`; this module fuses
the synchronised views into a single 3D track that `depth.py` can judge without
caring about camera geometry or occlusion on any one view.

Contract
--------
Input:  one or more time-synchronised `PoseSequence` views (same subject, same
        ``fps``), plus camera calibration.
Output: a single :class:`~whitelights.types.Pose3DSequence` in world
        coordinates (+z up), one fused frame per synchronised input frame, with
        a per-frame/per-keypoint confidence reflecting triangulation quality.

Requirements the implementation must honour:
  * Time alignment: views must be aligned to a common clock before
    triangulation (assume pre-synced for v2.0; genlock/offset handling later).
  * Robust triangulation: use all views that saw a keypoint with adequate
    confidence; degrade gracefully to a single view (no depth in z) or mark the
    keypoint low-confidence when views disagree or only one is available.
  * Confidence propagation: fused ``confidence`` should let downstream depth
    gating surface UNCERTAIN when reconstruction is poor.

Single-camera fallback (IMPLEMENTED): with exactly one view, the image plane is
lifted into world coordinates (x->x, y->z sign-flipped, world y=0) so the
depth-only v2.0 milestone can proceed before real multi-cam rigs exist. There is
no true depth signal in this mode.

TODO(ethan): multi-view triangulation. Needs a `CameraCalibration` type
(intrinsics + extrinsics per camera) and DLT / midpoint triangulation; currently
raises NotImplementedError for more than one view.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from .types import FrameKeypoints3D, Keypoint3D, Pose3DSequence, PoseSequence


def reconstruct_3d(views: Sequence[PoseSequence], calibration: Any | None = None) -> Pose3DSequence:
    """Fuse synchronised 2D camera views into one 3D pose track.

    Args:
        views: one or more time-synced single-camera pose tracks.
        calibration: per-camera intrinsics/extrinsics. Contract for this type is
            TODO(ethan); ``None`` implies the single-camera degenerate fallback.

    Returns:
        A `Pose3DSequence` in world coordinates (+z up).

    Raises:
        ValueError: if ``views`` is empty.
        NotImplementedError: for multi-view input (real triangulation is TODO).
    """
    if not views:
        raise ValueError("reconstruct_3d requires at least one camera view")
    if len(views) > 1:
        raise NotImplementedError("TODO(ethan): multi-view 3D triangulation not implemented")

    return _lift_single_view(views[0])


def _lift_single_view(view: PoseSequence) -> Pose3DSequence:
    """Degenerate single-camera fallback: lift a 2D view into 3D.

    Maps the image plane into world coordinates so `depth.py` can run before a
    calibrated multi-camera rig exists:

      * image ``x``      -> world ``x``   (unchanged)
      * image ``y``      -> world ``z``   sign-flipped (2D is +y-down, 3D is
                                          +z-up), so a keypoint lower in the
                                          image is lower in the world.
      * world ``y``      -> ``0.0``       (no depth information from one view)

    Undetected frames stay in the sequence with no keypoints (confidence 0), so
    the time base remains contiguous. There is genuinely no depth signal here;
    triangulation across views (real 3D) is a separate, later milestone.
    """
    frames: list[FrameKeypoints3D] = []
    for f in view.frames:
        keypoints = {
            name: Keypoint3D(name=name, x=kp.x, y=0.0, z=-kp.y, confidence=kp.confidence)
            for name, kp in f.keypoints.items()
        }
        frames.append(
            FrameKeypoints3D(
                frame_idx=f.frame_idx,
                time_s=f.time_s,
                keypoints=keypoints,
                confidence=f.subject_confidence,
            )
        )
    return Pose3DSequence(fps=view.fps, frames=frames, camera_ids=[view.camera_id])
