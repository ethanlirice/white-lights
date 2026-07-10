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

Single-camera fallback: with one view and no calibration, a caller may still
produce a degenerate `Pose3DSequence` (image plane lifted to z=0) so the
depth-only v2.0 milestone can proceed before real multi-cam rigs exist.

TODO(ethan): implement. Needs a `CameraCalibration` type (intrinsics +
extrinsics per camera) and DLT / midpoint triangulation.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from .types import Pose3DSequence, PoseSequence


def reconstruct_3d(views: Sequence[PoseSequence], calibration: Any | None = None) -> Pose3DSequence:
    """Fuse synchronised 2D camera views into one 3D pose track.

    Args:
        views: one or more time-synced single-camera pose tracks.
        calibration: per-camera intrinsics/extrinsics. Contract for this type is
            TODO(ethan); ``None`` implies the single-camera degenerate fallback.

    Returns:
        A `Pose3DSequence` in world coordinates (+z up).
    """
    raise NotImplementedError("TODO(ethan): multi-view 3D fusion not implemented")
