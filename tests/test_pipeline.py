"""Pipeline wiring test — pose runs, then the first stub halts the run.

Uses a fake `PoseEstimator` so no model weights or torch are required: it proves
the orchestration reaches the (stubbed) core and surfaces NotImplementedError,
exactly what the API maps to a 501.
"""

from __future__ import annotations

import pytest

from whitelights.pipeline import judge_video
from whitelights.pose import PoseEstimator
from whitelights.types import PoseSequence


class _FakeEstimator(PoseEstimator):
    """Returns an empty (but well-formed) pose track without loading a model."""

    def run_video(self, path, *, camera_id: str = "cam0") -> PoseSequence:
        return PoseSequence(camera_id=camera_id, fps=30.0, frames=[], source=str(path))


def test_pipeline_reaches_core_and_raises_not_implemented() -> None:
    est = _FakeEstimator()
    with pytest.raises(NotImplementedError):
        judge_video("fake.mp4", estimator=est)


def test_pipeline_requires_a_path() -> None:
    with pytest.raises(ValueError):
        judge_video([], estimator=_FakeEstimator())
