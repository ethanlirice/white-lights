"""Tests for `LiveJudge.process_frame`'s glue to the pose model.

`test_live.py` covers the tracker (`OnlineRepTracker`) directly — no pose
model involved. This file covers the one seam that touches
`estimator.model.predict(...)`, using a fake model in place of ultralytics
(same fake-`Results` shape `test_pose.py` uses) so it stays torch-free.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from whitelights.live import LiveJudge
from whitelights.pose import PoseEstimator


class _FakeKeypoints:
    def __init__(self, data: np.ndarray) -> None:
        self.data = data


class _FakeResult:
    """No `.boxes` — matches a live single-frame `predict()` call, which
    `result_to_frame` must handle by falling back to the confidence strategy."""

    def __init__(self, data: np.ndarray) -> None:
        self.keypoints = _FakeKeypoints(data)
        self.boxes = None


class _FakeModel:
    """Records every call to `.predict(...)` instead of running inference."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def predict(self, **kwargs: Any) -> list[_FakeResult]:
        self.calls.append(kwargs)
        return [_FakeResult(np.zeros((0, 17, 3)))]  # no person detected — fine, unused here


def _estimator_with_fake_model(*, device: str | None) -> tuple[PoseEstimator, _FakeModel]:
    estimator = PoseEstimator(device=device, conf=0.3)
    fake = _FakeModel()
    estimator._model = fake  # bypasses the lazy ultralytics import entirely
    return estimator, fake


def test_process_frame_forwards_the_configured_device() -> None:
    """Regression: process_frame used to omit `device=`, so a `PoseEstimator`
    configured for a non-default device would silently run on whatever
    ultralytics picks instead — while `run_video`/`run_frames` (the batch
    path) honoured it correctly. No real caller sets a non-default device
    today, which is exactly why this went unnoticed instead of unnoticed and
    harmless forever."""
    estimator, fake = _estimator_with_fake_model(device="mps")
    judge = LiveJudge(estimator)

    judge.process_frame(np.zeros((4, 4, 3), dtype=np.uint8))

    assert len(fake.calls) == 1
    assert fake.calls[0]["device"] == "mps"


def test_process_frame_forwards_conf_and_a_default_device_of_none() -> None:
    estimator, fake = _estimator_with_fake_model(device=None)
    judge = LiveJudge(estimator)

    judge.process_frame(np.zeros((4, 4, 3), dtype=np.uint8))

    call = fake.calls[0]
    assert call["conf"] == 0.3
    assert call["device"] is None
