"""Live webcam squat judge.

Real-time counterpart to the batch pipeline: instead of judging a finished clip,
it processes frames as they arrive from a camera and shows a live depth "light"
(red until the hip crease breaks the knee line, green once it does), a running
rep count, and a verdict when each rep completes.

Two pieces:
  * :class:`OnlineRepTracker` — the causal rep state machine. Unlike
    ``reps.segment_reps`` (which needs the whole clip's hip-travel range), this
    decides start / bottom / complete as frames arrive, using the lifter's thigh
    length as a per-frame scale reference (so thresholds are unit-invariant).
    Pure logic, no camera — unit-tested against synthetic frames.
  * :class:`LiveJudge` + :func:`main` — glue: webcam -> pose -> single-view 3D
    lift -> per-frame depth -> tracker -> overlay. OpenCV/ultralytics are
    imported lazily so importing this module (and testing the tracker) needs
    neither.

Run it::

    pip install -e ".[cv]"
    python -m whitelights.live            # default webcam
    python -m whitelights.live --camera 1 --model yolo11s-pose.pt

Press ESC or q to quit. Live judging covers depth + downward-movement; postural
and command faults are batch-only for now.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from enum import StrEnum

from pydantic import BaseModel

from .depth import DepthConfig, DepthFrameResult, judge_depth_frame
from .fusion import reconstruct_3d
from .pose import DEFAULT_MODEL, PoseEstimator, result_to_frame
from .types import Fault, FrameKeypoints, FrameKeypoints3D, PoseSequence, RepVerdict, Verdict

_HIP = ("left_hip", "right_hip")
_SIDES = ("left", "right")


class LiveState(StrEnum):
    STANDING = "STANDING"
    DESCENDING = "DESCENDING"
    ASCENDING = "ASCENDING"


class LiveConfig(BaseModel):
    """Thresholds for the online tracker, all as fractions of thigh length."""

    enter_fraction: float = 0.40  # descent below standing that starts a rep
    exit_fraction: float = 0.15  # rise back toward standing that ends a rep
    bottom_rise_fraction: float = 0.05  # rise above the running min that marks the bottom
    downward_movement_fraction: float = 0.05  # re-descent on the ascent -> fault
    min_thigh_length: float = 1e-6  # guard against degenerate scale


@dataclass
class LiveStatus:
    """Snapshot returned after each processed frame (drives the overlay)."""

    state: LiveState
    below_parallel: bool | None  # current-frame depth: True/False, or None if gated
    rep_count: int
    hip_z: float | None
    last_verdict: RepVerdict | None
    rep_completed: bool  # True only on the frame a rep finished


class OnlineRepTracker:
    """Causal rep detector: fed one (frame, depth) at a time, emits verdicts."""

    def __init__(self, config: LiveConfig | None = None) -> None:
        self.config = config or LiveConfig()
        self.state = LiveState.STANDING
        self._standing: float | None = None
        self._rep_count = 0
        self._last_verdict: RepVerdict | None = None
        self._reset_rep()

    def _reset_rep(self) -> None:
        self._start_frame = 0
        self._start_time = 0.0
        self._min_hip = math.inf
        self._ascent_peak = -math.inf
        self._reached_below = False
        self._had_confident = False
        self._best_margin: float | None = None
        self._best_conf = 0.0
        self._downward = False

    def update(self, frame: FrameKeypoints3D, depth: DepthFrameResult) -> LiveStatus:
        hip_z = _hip_z(frame)
        thigh = _thigh_length(frame)
        below = None if depth.gated else depth.is_below_parallel
        completed = False

        if hip_z is None or thigh is None or thigh < self.config.min_thigh_length:
            return self._status(below, hip_z, completed)

        if self._standing is None:
            self._standing = hip_z

        if self.state == LiveState.STANDING:
            self._standing = max(self._standing, hip_z)
            if hip_z < self._standing - self.config.enter_fraction * thigh:
                self._begin_rep(frame)
                self._accumulate(frame, depth, hip_z)
                self.state = LiveState.DESCENDING
        elif self.state == LiveState.DESCENDING:
            self._accumulate(frame, depth, hip_z)
            if hip_z > self._min_hip + self.config.bottom_rise_fraction * thigh:
                self.state = LiveState.ASCENDING
                self._ascent_peak = hip_z
        elif self.state == LiveState.ASCENDING:
            self._accumulate(frame, depth, hip_z)
            if hip_z < self._ascent_peak - self.config.downward_movement_fraction * thigh:
                self._downward = True
            self._ascent_peak = max(self._ascent_peak, hip_z)
            if hip_z >= self._standing - self.config.exit_fraction * thigh:
                self._last_verdict = self._finalize(frame)
                self._rep_count += 1
                completed = True
                self.state = LiveState.STANDING
                self._standing = hip_z

        return self._status(below, hip_z, completed)

    def _begin_rep(self, frame: FrameKeypoints3D) -> None:
        self._reset_rep()
        self._start_frame = frame.frame_idx
        self._start_time = frame.time_s

    def _accumulate(self, frame: FrameKeypoints3D, depth: DepthFrameResult, hip_z: float) -> None:
        self._min_hip = min(self._min_hip, hip_z)
        if not depth.gated and depth.depth_margin is not None:
            self._had_confident = True
            if depth.is_below_parallel:
                self._reached_below = True
            if self._best_margin is None or depth.depth_margin > self._best_margin:
                self._best_margin = depth.depth_margin
                self._best_conf = depth.confidence

    def _finalize(self, frame: FrameKeypoints3D) -> RepVerdict:
        faults: list[Fault] = []
        if self._had_confident and not self._reached_below:
            faults.append(Fault.INSUFFICIENT_DEPTH)
        if self._downward:
            faults.append(Fault.DOWNWARD_MOVEMENT)

        if faults:
            verdict = Verdict.NO_LIFT
        elif not self._had_confident:
            verdict = Verdict.UNCERTAIN
        else:
            verdict = Verdict.GOOD

        return RepVerdict(
            rep_index=self._rep_count,
            verdict=verdict,
            confidence=self._best_conf,
            faults=faults,
            depth_margin=self._best_margin,
            start_frame=self._start_frame,
            end_frame=frame.frame_idx,
            start_time_s=self._start_time,
            end_time_s=frame.time_s,
        )

    def _status(self, below: bool | None, hip_z: float | None, completed: bool) -> LiveStatus:
        return LiveStatus(
            state=self.state,
            below_parallel=below,
            rep_count=self._rep_count,
            hip_z=hip_z,
            last_verdict=self._last_verdict,
            rep_completed=completed,
        )


def _hip_z(frame: FrameKeypoints3D) -> float | None:
    zs = [kp.z for name in _HIP if (kp := frame.get(name)) is not None]
    return sum(zs) / len(zs) if zs else None


def _thigh_length(frame: FrameKeypoints3D) -> float | None:
    lengths: list[float] = []
    for side in _SIDES:
        hip = frame.get(f"{side}_hip")
        knee = frame.get(f"{side}_knee")
        if hip is not None and knee is not None:
            lengths.append(math.dist((hip.x, hip.y, hip.z), (knee.x, knee.y, knee.z)))
    return sum(lengths) / len(lengths) if lengths else None


def lift_frame_to_3d(frame2d: FrameKeypoints, *, fps: float, camera_id: str = "cam0"):
    """Single-view 2D->3D lift for one frame (reuses the fusion fallback)."""
    seq = PoseSequence(camera_id=camera_id, fps=fps, frames=[frame2d])
    return reconstruct_3d([seq]).frames[0]


class LiveJudge:
    """Ties the per-frame pipeline together for a live stream."""

    def __init__(
        self,
        estimator: PoseEstimator | None = None,
        *,
        fps: float = 30.0,
        depth_config: DepthConfig | None = None,
        live_config: LiveConfig | None = None,
    ) -> None:
        self.estimator = estimator or PoseEstimator()
        self.fps = fps
        self.depth_config = depth_config or DepthConfig()
        self.tracker = OnlineRepTracker(live_config)
        self._frame_idx = 0

    def process_frame(self, bgr_frame) -> tuple[FrameKeypoints, DepthFrameResult, LiveStatus]:
        result = self.estimator.model.predict(
            source=bgr_frame, conf=self.estimator.conf, verbose=False
        )
        frame2d = result_to_frame(result[0], self._frame_idx, self.fps, self.estimator.subject)
        frame3d = lift_frame_to_3d(frame2d, fps=self.fps)
        depth = judge_depth_frame(frame3d, self.depth_config)
        status = self.tracker.update(frame3d, depth)
        self._frame_idx += 1
        return frame2d, depth, status


# ---------------------------------------------------------------------------
# Webcam demo (OpenCV) — imported lazily, not covered by unit tests
# ---------------------------------------------------------------------------

_SKELETON = [
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
]


def _draw_overlay(img, frame2d: FrameKeypoints, status: LiveStatus, conf: float) -> None:
    import cv2

    # Light: green below parallel, red above, grey when unknown.
    color = (128, 128, 128)
    if status.below_parallel is True:
        color = (0, 200, 0)
    elif status.below_parallel is False:
        color = (0, 0, 220)
    cv2.circle(img, (40, 40), 22, color, -1)

    for a, b in _SKELETON:
        ka, kb = frame2d.get(a), frame2d.get(b)
        if ka and kb and ka.confidence >= conf and kb.confidence >= conf:
            cv2.line(img, (int(ka.x), int(ka.y)), (int(kb.x), int(kb.y)), (240, 240, 240), 2)
    for kp in frame2d.keypoints.values():
        if kp.confidence >= conf:
            cv2.circle(img, (int(kp.x), int(kp.y)), 4, (0, 200, 255), -1)

    lines = [f"{status.state}   reps: {status.rep_count}"]
    if status.last_verdict is not None:
        v = status.last_verdict
        tag = v.verdict.value + ("  " + ",".join(f.value for f in v.faults) if v.faults else "")
        lines.append(f"last: {tag}")
    y = 80
    for text in lines:
        cv2.putText(img, text, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        y += 30


def main() -> None:
    parser = argparse.ArgumentParser(description="White Lights — live webcam squat judge")
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default 0)")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="YOLO11-pose weights")
    parser.add_argument("--conf", type=float, default=0.25, help="Detection confidence")
    args = parser.parse_args()

    import cv2

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"Could not open camera {args.camera}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    judge = LiveJudge(PoseEstimator(model_path=args.model, conf=args.conf), fps=fps)
    print("White Lights live — press ESC or q to quit.")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame2d, _depth, status = judge.process_frame(frame)
            _draw_overlay(frame, frame2d, status, args.conf)
            if status.rep_completed and status.last_verdict is not None:
                print(f"rep {status.last_verdict.rep_index}: {status.last_verdict.verdict.value}")
            cv2.imshow("White Lights — live", frame)
            if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
