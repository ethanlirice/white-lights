"""Contract tests for `smoothing.smooth_sequence`."""

from __future__ import annotations

from whitelights.smoothing import SmoothingConfig, smooth_sequence
from whitelights.types import FrameKeypoints, Keypoint2D, PoseSequence


def _seq(
    frames: list[FrameKeypoints], *, fps: float = 30.0, camera_id: str = "camA"
) -> PoseSequence:
    return PoseSequence(camera_id=camera_id, fps=fps, frames=frames)


def _hip_frame(idx: int, x: float | None, y: float | None, conf: float) -> FrameKeypoints:
    """A frame with a single left_hip, or an empty frame when x/y is None."""
    kps = {}
    if x is not None and y is not None:
        kps = {"left_hip": Keypoint2D(name="left_hip", x=x, y=y, confidence=conf)}
    return FrameKeypoints(
        frame_idx=idx,
        time_s=idx / 30.0,
        keypoints=kps,
        detected=bool(kps),
        subject_confidence=conf if kps else 0.0,
    )


def test_preserves_time_base(noisy_pose_2d) -> None:
    out = smooth_sequence(noisy_pose_2d)
    assert len(out.frames) == len(noisy_pose_2d.frames)
    assert out.fps == noisy_pose_2d.fps
    assert out.camera_id == noisy_pose_2d.camera_id
    assert [f.time_s for f in out.frames] == [f.time_s for f in noisy_pose_2d.frames]


def test_fills_short_gap(noisy_pose_2d) -> None:
    out = smooth_sequence(noisy_pose_2d)
    # frames 18-21 were dropouts; a short gap should be bridged.
    for i in range(18, 22):
        assert out.frames[i].get("left_hip") is not None


def test_short_gap_is_linearly_interpolated() -> None:
    seq = _seq(
        [
            _hip_frame(0, 100.0, 200.0, 0.9),
            _hip_frame(1, None, None, 0.0),  # gap
            _hip_frame(2, 300.0, 400.0, 0.9),
        ]
    )
    out = smooth_sequence(seq)
    mid = out.frames[1].get("left_hip")
    assert mid is not None
    assert mid.x == 200.0  # midpoint of 100 and 300
    assert mid.y == 300.0


def test_long_gap_is_not_filled() -> None:
    frames = [_hip_frame(0, 100.0, 200.0, 0.9)]
    frames += [_hip_frame(i, None, None, 0.0) for i in range(1, 9)]  # 8-frame gap
    frames.append(_hip_frame(9, 300.0, 400.0, 0.9))
    out = smooth_sequence(_seq(frames), config=SmoothingConfig(max_gap_frames=5))
    for i in range(1, 9):
        assert out.frames[i].get("left_hip") is None  # too long to bridge


def test_low_confidence_sample_is_gated() -> None:
    # A lone low-confidence sample with no anchors is dropped, not trusted.
    seq = _seq([_hip_frame(0, 100.0, 200.0, 0.1)])
    out = smooth_sequence(seq, config=SmoothingConfig(min_confidence=0.3))
    assert out.frames[0].get("left_hip") is None
    assert out.frames[0].detected is False
