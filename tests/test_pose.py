"""Tests for the pose wrapper's pure helpers (no ultralytics / torch needed)."""

from __future__ import annotations

import numpy as np

from whitelights.pose import (
    empty_frame,
    frame_from_person,
    result_to_frame,
    select_subject_index,
)
from whitelights.types import COCO_KEYPOINT_NAMES


class _FakeKeypoints:
    def __init__(self, data: np.ndarray) -> None:
        self.data = data


class _FakeBoxes:
    def __init__(self, xyxy: np.ndarray) -> None:
        self.xyxy = xyxy


class _FakeResult:
    def __init__(self, data: np.ndarray, xyxy: np.ndarray | None = None) -> None:
        self.keypoints = _FakeKeypoints(data)
        self.boxes = _FakeBoxes(xyxy) if xyxy is not None else None


def _person(x: float, conf: float) -> np.ndarray:
    """A (17, 3) person with every keypoint at (x, 0) and the given confidence."""
    p = np.zeros((17, 3), dtype=float)
    p[:, 0] = x
    p[:, 2] = conf
    return p


def test_select_subject_prefers_largest_box() -> None:
    data = np.stack([_person(10, 0.9), _person(20, 0.5)])
    boxes = np.array([[0, 0, 5, 5], [0, 0, 50, 50]], dtype=float)  # person 1 much larger
    assert select_subject_index(data, boxes, "largest") == 1


def test_select_subject_confidence_strategy() -> None:
    data = np.stack([_person(10, 0.9), _person(20, 0.5)])
    assert select_subject_index(data, None, "confidence") == 0


def test_select_subject_none_when_empty() -> None:
    assert select_subject_index(np.zeros((0, 17, 3)), None) is None


def test_frame_from_person_maps_names_and_confidence() -> None:
    frame = frame_from_person(_person(42.0, 0.8), frame_idx=3, fps=30.0)
    assert frame.detected is True
    assert frame.frame_idx == 3
    assert frame.time_s == 3 / 30.0
    assert set(frame.keypoints) == set(COCO_KEYPOINT_NAMES)
    assert frame.get("nose").x == 42.0
    assert abs(frame.subject_confidence - 0.8) < 1e-9


def test_result_to_frame_selects_largest_and_converts() -> None:
    data = np.stack([_person(10, 0.9), _person(20, 0.5)])
    boxes = np.array([[0, 0, 5, 5], [0, 0, 50, 50]], dtype=float)
    frame = result_to_frame(_FakeResult(data, boxes), frame_idx=0, fps=30.0, strategy="largest")
    assert frame.detected is True
    assert frame.get("left_hip").x == 20.0  # the larger person was chosen


def test_result_to_frame_handles_no_detection() -> None:
    frame = result_to_frame(_FakeResult(np.zeros((0, 17, 3))), frame_idx=7, fps=30.0)
    assert frame.detected is False
    assert frame.keypoints == {}
    assert frame.frame_idx == 7


def test_empty_frame_time_base() -> None:
    frame = empty_frame(frame_idx=15, fps=30.0)
    assert frame.detected is False
    assert frame.time_s == 0.5
