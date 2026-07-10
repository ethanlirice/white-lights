"""Tests for the single-camera fallback in `fusion.reconstruct_3d`.

Multi-view triangulation is still a stub (raises); these cover the degenerate
one-view lift used by the depth-only v2.0 path.
"""

from __future__ import annotations

import pytest

from whitelights.fusion import reconstruct_3d
from whitelights.types import (
    FrameKeypoints,
    Keypoint2D,
    Pose3DSequence,
    PoseSequence,
)


def _view(*frames: FrameKeypoints, camera_id: str = "camA", fps: float = 30.0) -> PoseSequence:
    return PoseSequence(camera_id=camera_id, fps=fps, frames=list(frames))


def _frame(idx: int, kps: dict[str, tuple[float, float, float]], *, conf: float) -> FrameKeypoints:
    """Build a 2D frame from name -> (x, y, confidence)."""
    keypoints = {
        name: Keypoint2D(name=name, x=x, y=y, confidence=c) for name, (x, y, c) in kps.items()
    }
    return FrameKeypoints(
        frame_idx=idx,
        time_s=idx / 30.0,
        keypoints=keypoints,
        detected=bool(keypoints),
        subject_confidence=conf,
    )


def test_lift_maps_axes_and_preserves_metadata() -> None:
    view = _view(
        _frame(0, {"left_hip": (100.0, 300.0, 0.9)}, conf=0.9),
        camera_id="camA",
        fps=30.0,
    )
    out = reconstruct_3d([view])

    assert isinstance(out, Pose3DSequence)
    assert out.fps == 30.0
    assert out.camera_ids == ["camA"]
    assert len(out.frames) == 1

    frame = out.frames[0]
    assert frame.frame_idx == 0
    assert frame.time_s == 0.0
    assert frame.confidence == 0.9

    kp = frame.get("left_hip")
    assert kp.x == 100.0  # x unchanged
    assert kp.y == 0.0  # no depth from a single view
    assert kp.z == -300.0  # image y -> world z, sign-flipped
    assert kp.confidence == 0.9  # per-keypoint confidence preserved


def test_below_parallel_sign_is_consistent_with_depth_convention() -> None:
    # Hip lower in the image (larger y) than the knee == below parallel.
    view = _view(
        _frame(
            0,
            {"left_hip": (0.0, 400.0, 0.9), "left_knee": (0.0, 350.0, 0.9)},
            conf=0.9,
        )
    )
    frame = reconstruct_3d([view]).frames[0]
    hip_z = frame.get("left_hip").z
    knee_z = frame.get("left_knee").z

    assert hip_z < knee_z  # hip is lower in world coords
    assert knee_z - hip_z > 0  # depth_margin convention: positive == below parallel


def test_undetected_frame_stays_empty_and_keeps_time_base() -> None:
    view = _view(
        _frame(0, {"left_hip": (100.0, 300.0, 0.9)}, conf=0.9),
        _frame(1, {}, conf=0.0),  # dropout
        _frame(2, {"left_hip": (101.0, 305.0, 0.8)}, conf=0.8),
    )
    out = reconstruct_3d([view])

    assert [f.frame_idx for f in out.frames] == [0, 1, 2]
    assert out.frames[1].keypoints == {}
    assert out.frames[1].confidence == 0.0
    assert out.frames[1].get("left_hip") is None


def test_multi_view_raises_not_implemented() -> None:
    view = _view(_frame(0, {"left_hip": (1.0, 2.0, 0.9)}, conf=0.9))
    with pytest.raises(NotImplementedError):
        reconstruct_3d([view, view])


def test_empty_views_raises_value_error() -> None:
    with pytest.raises(ValueError):
        reconstruct_3d([])
