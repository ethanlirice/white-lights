"""End-to-end pipeline tests using a fake `PoseEstimator` (no weights / torch).

With smoothing, fusion (single-view), depth, and reps implemented, a single-view
clip now runs the whole pipeline and returns real per-rep verdicts. Multi-view
still raises NotImplementedError at the fusion stage.
"""

from __future__ import annotations

import pytest

from whitelights.pipeline import judge_video
from whitelights.pose import PoseEstimator
from whitelights.types import FrameKeypoints, Keypoint2D, PoseSequence, Verdict


def _synthetic_2d_squat(n: int = 60, fps: float = 30.0) -> PoseSequence:
    """A 2D (image-coord, +y down) good-depth squat: hips descend below the knee.

    knee_y is fixed; hip_y sweeps 100 -> 350 -> 100. At the bottom hip_y (350) is
    below knee_y (300), i.e. below parallel once lifted into 3D.
    """
    half = n // 2
    down = [100.0 + 250.0 * (k / half) for k in range(half)]
    up = [350.0 - 250.0 * (k / (n - half)) for k in range(n - half)]
    hip_y = down + up

    frames: list[FrameKeypoints] = []
    for i, hy in enumerate(hip_y):
        kps = {
            "left_hip": Keypoint2D(name="left_hip", x=-10.0, y=hy, confidence=0.9),
            "right_hip": Keypoint2D(name="right_hip", x=10.0, y=hy, confidence=0.9),
            "left_knee": Keypoint2D(name="left_knee", x=-10.0, y=300.0, confidence=0.9),
            "right_knee": Keypoint2D(name="right_knee", x=10.0, y=300.0, confidence=0.9),
        }
        frames.append(
            FrameKeypoints(
                frame_idx=i, time_s=i / fps, keypoints=kps, detected=True, subject_confidence=0.9
            )
        )
    return PoseSequence(camera_id="cam0", fps=fps, frames=frames)


class _FakeEstimator(PoseEstimator):
    """Returns a synthetic squat track without loading a model."""

    def run_video(self, path, *, camera_id: str = "cam0") -> PoseSequence:
        seq = _synthetic_2d_squat()
        seq.camera_id = camera_id
        seq.source = str(path)
        return seq


def test_single_view_pipeline_returns_good_verdict() -> None:
    result = judge_video("fake.mp4", estimator=_FakeEstimator())

    assert result.fps == 30.0
    assert result.frame_count == 60
    assert result.camera_ids == ["cam0"]
    assert len(result.reps) == 1

    rep = result.reps[0]
    assert rep.verdict == Verdict.GOOD
    assert rep.faults == []
    assert rep.depth_margin > 0
    assert result.processing_ms >= 0.0


def test_multi_view_raises_not_implemented() -> None:
    with pytest.raises(NotImplementedError):
        judge_video(["a.mp4", "b.mp4"], estimator=_FakeEstimator())


def test_pipeline_requires_a_path() -> None:
    with pytest.raises(ValueError):
        judge_video([], estimator=_FakeEstimator())
