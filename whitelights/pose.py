"""YOLO11-pose wrapper — the one fully implemented module.

Loads a YOLO11-pose model and runs it over a video file or an in-memory frame
stream, returning a typed :class:`~whitelights.types.PoseSequence` (per-frame,
per-keypoint, with confidences) for a single camera view.

The heavy dependencies (`ultralytics`, `opencv-python`) are imported lazily so
that importing this module — and unit-testing the pure parsing helpers below —
does not require torch or model weights. Install them with the ``cv`` extra::

    pip install -e ".[cv]"

Model weights (``yolo11n-pose.pt`` by default) are auto-downloaded by
ultralytics on first use.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Literal

import numpy as np

from .types import COCO_KEYPOINT_NAMES, FrameKeypoints, Keypoint2D, PoseSequence

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "yolo11n-pose.pt"
DEFAULT_FPS = 30.0

SubjectStrategy = Literal["largest", "confidence"]


# ---------------------------------------------------------------------------
# Pure helpers (no ultralytics / torch — unit-tested directly)
# ---------------------------------------------------------------------------


def select_subject_index(
    keypoints_xyc: np.ndarray,
    boxes_xyxy: np.ndarray | None = None,
    strategy: SubjectStrategy = "largest",
) -> int | None:
    """Choose which detected person is the lifter.

    Args:
        keypoints_xyc: ``(n_persons, 17, 3)`` array of ``(x, y, confidence)``.
        boxes_xyxy: optional ``(n_persons, 4)`` bounding boxes for the
            ``"largest"`` strategy. Falls back to ``"confidence"`` if absent.
        strategy: ``"largest"`` picks the biggest bounding box (the lifter is
            usually closest to camera); ``"confidence"`` picks the highest mean
            keypoint confidence.

    Returns:
        The chosen person index, or ``None`` if no person was detected.
    """
    n = keypoints_xyc.shape[0]
    if n == 0:
        return None
    if n == 1:
        return 0
    if strategy == "largest" and boxes_xyxy is not None and len(boxes_xyxy) == n:
        widths = boxes_xyxy[:, 2] - boxes_xyxy[:, 0]
        heights = boxes_xyxy[:, 3] - boxes_xyxy[:, 1]
        return int(np.argmax(widths * heights))
    return int(np.argmax(keypoints_xyc[:, :, 2].mean(axis=1)))


def frame_from_person(person_xyc: np.ndarray, frame_idx: int, fps: float) -> FrameKeypoints:
    """Convert one person's ``(17, 3)`` keypoint array into a `FrameKeypoints`."""
    keypoints: dict[str, Keypoint2D] = {}
    confidences: list[float] = []
    for i, name in enumerate(COCO_KEYPOINT_NAMES):
        x, y, c = person_xyc[i]
        keypoints[name] = Keypoint2D(name=name, x=float(x), y=float(y), confidence=float(c))
        confidences.append(float(c))
    return FrameKeypoints(
        frame_idx=frame_idx,
        time_s=frame_idx / fps if fps else 0.0,
        keypoints=keypoints,
        detected=True,
        subject_confidence=float(np.mean(confidences)) if confidences else 0.0,
    )


def empty_frame(frame_idx: int, fps: float) -> FrameKeypoints:
    """A frame where no subject was detected (keeps the time base contiguous)."""
    return FrameKeypoints(
        frame_idx=frame_idx,
        time_s=frame_idx / fps if fps else 0.0,
        keypoints={},
        detected=False,
        subject_confidence=0.0,
    )


def result_to_frame(
    result: Any, frame_idx: int, fps: float, strategy: SubjectStrategy = "largest"
) -> FrameKeypoints:
    """Extract a `FrameKeypoints` from a single ultralytics ``Results`` object.

    Kept tolerant of empty detections and duck-typed so it can be exercised with
    lightweight fakes in tests (see ``tests/test_pose.py``).
    """
    kp = getattr(result, "keypoints", None)
    data = getattr(kp, "data", None)
    if kp is None or data is None or len(data) == 0:
        return empty_frame(frame_idx, fps)

    keypoints_xyc = _to_numpy(data)  # (n, 17, 3)

    boxes_xyxy = None
    boxes = getattr(result, "boxes", None)
    box_xyxy = getattr(boxes, "xyxy", None)
    if box_xyxy is not None and len(box_xyxy) > 0:
        boxes_xyxy = _to_numpy(box_xyxy)

    idx = select_subject_index(keypoints_xyc, boxes_xyxy, strategy)
    if idx is None:
        return empty_frame(frame_idx, fps)
    return frame_from_person(keypoints_xyc[idx], frame_idx, fps)


def _to_numpy(x: Any) -> np.ndarray:
    """Accept a torch tensor or anything array-like and return a numpy array."""
    cpu = getattr(x, "cpu", None)
    if cpu is not None:
        x = cpu()
    numpy = getattr(x, "numpy", None)
    if numpy is not None:
        return numpy()
    return np.asarray(x)


def read_fps(path: str | Path) -> float:
    """Read the frame rate of a video file, defaulting sensibly if unavailable."""
    import cv2  # lazy: only needed when actually decoding video

    cap = cv2.VideoCapture(str(path))
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
    finally:
        cap.release()
    return float(fps) if fps and fps > 0 else DEFAULT_FPS


# ---------------------------------------------------------------------------
# Estimator
# ---------------------------------------------------------------------------


class PoseEstimator:
    """Thin, reusable wrapper around a YOLO11-pose model.

    The model is loaded lazily on first inference so constructing an estimator
    is cheap (and import-safe without the ``cv`` extra installed).
    """

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL,
        *,
        device: str | None = None,
        conf: float = 0.25,
        subject: SubjectStrategy = "largest",
    ) -> None:
        self.model_path = model_path
        self.device = device
        self.conf = conf
        self.subject = subject
        self._model: Any | None = None

    @property
    def model(self) -> Any:
        if self._model is None:
            from ultralytics import YOLO  # lazy import of the heavy stack

            logger.info("Loading YOLO11-pose model: %s", self.model_path)
            self._model = YOLO(self.model_path)
        return self._model

    def run_video(self, path: str | Path, *, camera_id: str = "cam0") -> PoseSequence:
        """Run pose estimation over every frame of a video file."""
        fps = read_fps(path)
        results = self.model.predict(
            source=str(path),
            stream=True,
            conf=self.conf,
            device=self.device,
            verbose=False,
        )
        frames = [result_to_frame(r, i, fps, self.subject) for i, r in enumerate(results)]
        return PoseSequence(camera_id=camera_id, fps=fps, frames=frames, source=str(path))

    def run_frames(
        self,
        frames: Iterable[np.ndarray],
        *,
        fps: float = DEFAULT_FPS,
        camera_id: str = "cam0",
    ) -> PoseSequence:
        """Run pose estimation over an in-memory stream of BGR frames."""
        out: list[FrameKeypoints] = []
        for i, frame in enumerate(frames):
            result = self.model.predict(
                source=frame, conf=self.conf, device=self.device, verbose=False
            )[0]
            out.append(result_to_frame(result, i, fps, self.subject))
        return PoseSequence(camera_id=camera_id, fps=fps, frames=out, source=None)
