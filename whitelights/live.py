"""Live webcam squat judge.

Real-time counterpart to the batch pipeline: it processes frames as they arrive
from a camera and shows a live depth "light" (red until the hip crease breaks the
knee line, green once it does), a running rep count, a verdict when each rep
completes, and — importantly — *why* it is doing what it is doing.

Pieces:
  * :class:`OnlineRepTracker` — the causal rep state machine. It is deliberately
    conservative so it does not phantom-count reps on noisy webcam pose:
      - works off confidence-gated, smoothed keypoints (see `LiveJudge`);
      - keeps an *adaptive* standing reference (re-baselined only while the
        lifter is actually still), scaled by the standing thigh length so
        thresholds are unit-invariant;
      - requires a real descent (deep enough + long enough, returning to
        standing) before counting — shallow bobs and single-frame jitter are
        discarded, not counted.
    Pure logic, no camera. Unit-tested against synthetic frames.
  * :class:`CompetitionTracker` — the referee-command single-attempt judge.
  * :class:`LiveJudge` — per-frame glue: pose -> smoothing -> single-view 3D
    lift -> per-frame depth -> tracker.

Judging logic only: the OpenCV webcam demo and its drawing code live in
`whitelights.cli`, which is where ``python -m whitelights.cli`` runs from.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import StrEnum

from pydantic import BaseModel, Field

from .depth import DepthConfig, DepthFrameResult, judge_depth_frame
from .filters import StreamingKeypointSmoother
from .fusion import reconstruct_3d
from .pose import PoseEstimator, result_to_frame
from .posture import FootDriftMonitor, PostureConfig, is_locked_out
from .tracking import HoldTimer, MotionTracker, decide, mean_height, segment_length
from .types import (
    Fault,
    FrameKeypoints,
    FrameKeypoints3D,
    LiveStatus,
    PoseSequence,
    RepVerdict,
)

# Re-exported for the lift modules and the API, which have always imported it
# from here; it now lives in `types` because every tracker shares it.
__all__ = ["LiveStatus"]

_HIP = ("left_hip", "right_hip")
_THIGH = ("hip", "knee")


class LiveState(StrEnum):
    STANDING = "STANDING"
    DESCENDING = "DESCENDING"
    ASCENDING = "ASCENDING"


class LiveConfig(BaseModel):
    """Thresholds for the online tracker. Distances are fractions of thigh length."""

    min_confidence: float = 0.5  # per-keypoint confidence to trust a hip/knee
    enter_fraction: float = 0.30  # drop below standing (with downward motion) to start
    exit_fraction: float = 0.15  # rise back toward standing to end the rep
    bottom_rise_fraction: float = 0.05  # rise above the running min that marks the bottom
    downward_movement_fraction: float = 0.05  # re-descent on the ascent -> fault
    min_rep_depth_fraction: float = 0.50  # total descent required to count as a real rep
    min_rep_duration_s: float = 0.40  # minimum rep length; rejects flickers
    still_velocity_fraction: float = 0.50  # |hip velocity| (per s) below this == "still"
    standing_ema: float = 0.20  # how fast the standing reference re-baselines when still
    max_lost_frames: int = 8  # dropouts before an in-progress rep is abandoned
    posture: PostureConfig = Field(default_factory=PostureConfig)


@dataclass
class _Candidate:
    """Accumulator for a rep in progress."""

    start_frame: int = 0
    start_time: float = 0.0
    min_hip: float = math.inf
    ascent_peak: float = -math.inf
    reached_below: bool = False
    had_confident: bool = False
    best_margin: float | None = None
    best_conf: float = 0.0
    downward: bool = False
    lost: int = 0
    faults: list[Fault] = field(default_factory=list)


class OnlineRepTracker:
    """Causal rep detector: fed one (frame, depth) at a time, emits verdicts."""

    def __init__(self, config: LiveConfig | None = None) -> None:
        self.config = config or LiveConfig()
        self.reset()

    def reset(self) -> None:
        self.state = LiveState.STANDING
        self._standing_hip: float | None = None
        self._standing_thigh: float | None = None
        self._motion = MotionTracker(self.config.still_velocity_fraction)
        self._rep_count = 0
        self._last_verdict: RepVerdict | None = None
        self._cand = _Candidate()
        self._feet = FootDriftMonitor(self.config.posture)

    def update(self, frame: FrameKeypoints3D, depth: DepthFrameResult) -> LiveStatus:
        c = self.config
        hip = mean_height(frame, _HIP, c.min_confidence)
        thigh = segment_length(frame, _THIGH, c.min_confidence)
        below = None if depth.gated else depth.is_below_parallel
        margin = None if depth.gated else depth.depth_margin

        # No reliable pose this frame: hold state, count dropouts during a rep.
        if hip is None or thigh is None or thigh <= 0:
            if self.state != LiveState.STANDING:
                self._cand.lost += 1
                if self._cand.lost > c.max_lost_frames:
                    self.state = LiveState.STANDING
                    return self._status(below, margin, hip, "lost the lifter — reset")
            return self._status(below, margin, hip, "waiting for a clear view of hips + knees")

        if self._standing_hip is None:
            self._standing_hip, self._standing_thigh = hip, thigh

        vel = self._motion.observe(hip, frame.time_s)

        scale = self._standing_thigh or thigh
        still = self._motion.is_still(scale)
        completed = False
        note = ""

        if self.state == LiveState.STANDING:
            if still:  # re-baseline the standing reference only when actually still
                a = c.standing_ema
                self._standing_hip = (1 - a) * self._standing_hip + a * hip
                self._standing_thigh = (1 - a) * (self._standing_thigh or thigh) + a * thigh
                scale = self._standing_thigh
            enter = self._standing_hip - c.enter_fraction * scale
            if hip < enter and vel < 0:
                self._begin_rep(frame, hip)
                self.state = LiveState.DESCENDING
                note = "descending…"
            else:
                note = "standing — watching for a descent"
        elif self.state == LiveState.DESCENDING:
            self._accumulate(frame, depth, hip)
            if hip > self._cand.min_hip + c.bottom_rise_fraction * scale:
                self.state = LiveState.ASCENDING
                self._cand.ascent_peak = hip
                note = "out of the hole, standing up…"
            else:
                note = f"descending… depth {self._descent_frac(hip, scale) * 100:.0f}% of a rep"
        elif self.state == LiveState.ASCENDING:
            self._accumulate(frame, depth, hip)
            if hip < self._cand.ascent_peak - c.downward_movement_fraction * scale:
                self._cand.downward = True
            self._cand.ascent_peak = max(self._cand.ascent_peak, hip)
            if hip >= self._standing_hip - c.exit_fraction * scale:  # back to lockout
                completed, note = self._complete(frame, scale)
                self.state = LiveState.STANDING
                self._standing_hip = hip
            else:
                note = "standing up…"

        return self._status(below, margin, hip, note, completed=completed, scale=scale)

    # -- rep lifecycle -------------------------------------------------------

    def _begin_rep(self, frame: FrameKeypoints3D, hip: float) -> None:
        self._cand = _Candidate(start_frame=frame.frame_idx, start_time=frame.time_s, min_hip=hip)
        self._feet.reset()
        self._feet.observe(frame)

    def _accumulate(self, frame: FrameKeypoints3D, depth: DepthFrameResult, hip: float) -> None:
        self._cand.min_hip = min(self._cand.min_hip, hip)
        self._cand.lost = 0
        self._feet.observe(frame)
        if not depth.gated and depth.depth_margin is not None:
            self._cand.had_confident = True
            if depth.is_below_parallel:
                self._cand.reached_below = True
            if self._cand.best_margin is None or depth.depth_margin > self._cand.best_margin:
                self._cand.best_margin = depth.depth_margin
                self._cand.best_conf = depth.confidence

    def _complete(self, frame: FrameKeypoints3D, scale: float) -> tuple[bool, str]:
        c = self.config
        depth_drop = self._standing_hip - self._cand.min_hip
        duration = frame.time_s - self._cand.start_time
        if depth_drop < c.min_rep_depth_fraction * scale or duration < c.min_rep_duration_s:
            return False, "movement too shallow/short — not a rep"
        self._last_verdict = self._finalize(frame)
        self._rep_count += 1
        return True, f"REP {self._rep_count}: {self._verdict_label(self._last_verdict)}"

    def _finalize(self, frame: FrameKeypoints3D) -> RepVerdict:
        cand = self._cand
        faults: list[Fault] = []
        if cand.had_confident and not cand.reached_below:
            faults.append(Fault.INSUFFICIENT_DEPTH)
        if cand.downward:
            faults.append(Fault.DOWNWARD_MOVEMENT)
        if self._feet.moved():
            faults.append(Fault.FOOT_MOVEMENT)

        verdict = decide(faults, uncertain=not cand.had_confident)

        return RepVerdict(
            rep_index=self._rep_count,
            verdict=verdict,
            confidence=cand.best_conf,
            faults=faults,
            depth_margin=cand.best_margin,
            start_frame=cand.start_frame,
            end_frame=frame.frame_idx,
            start_time_s=cand.start_time,
            end_time_s=frame.time_s,
        )

    # -- helpers -------------------------------------------------------------

    @staticmethod
    def _verdict_label(v: RepVerdict) -> str:
        return v.verdict.value + (" — " + ", ".join(f.value for f in v.faults) if v.faults else "")

    def _descent_frac(self, hip: float, scale: float) -> float:
        if self._standing_hip is None or scale <= 0:
            return 0.0
        return max(0.0, (self._standing_hip - hip) / scale)

    def _status(
        self,
        below: bool | None,
        margin: float | None,
        hip: float | None,
        note: str,
        *,
        completed: bool = False,
        scale: float | None = None,
    ) -> LiveStatus:
        frac = None
        if hip is not None and scale:
            frac = self._descent_frac(hip, scale)
        return LiveStatus(
            state=self.state,
            note=note,
            below_parallel=below,
            checkpoint=below,  # squat's key checkpoint == below parallel
            depth_margin=margin,
            hip_z=hip,
            standing_ref=self._standing_hip,
            descent_fraction=frac,
            rep_count=self._rep_count,
            last_verdict=self._last_verdict,
            rep_completed=completed,
        )


# ---------------------------------------------------------------------------
# Competition mode: the computer plays referee
# ---------------------------------------------------------------------------


class CompState(StrEnum):
    AWAIT_SETUP = "AWAIT_SETUP"  # waiting for a still, locked setup -> SQUAT
    SET = "SET"  # SQUAT given, waiting for the descent
    DESCENDING = "DESCENDING"
    ASCENDING = "ASCENDING"
    AWAIT_LOCKOUT = "AWAIT_LOCKOUT"  # waiting for a still, locked finish -> RACK
    DONE = "DONE"


class CompetitionConfig(BaseModel):
    """Thresholds for the competition (referee-command) judge."""

    min_confidence: float = 0.5
    enter_fraction: float = 0.30
    exit_fraction: float = 0.15
    bottom_rise_fraction: float = 0.05
    downward_movement_fraction: float = 0.05
    still_velocity_fraction: float = 0.40  # |hip velocity| (per s) below this == still
    setup_hold_s: float = 0.60  # still + locked hold before SQUAT
    lockout_hold_s: float = 0.60  # still hold at the top before RACK
    max_lockout_wait_s: float = 6.0  # give RACK anyway after this long at the top
    posture: PostureConfig = Field(default_factory=PostureConfig)


class CompetitionTracker:
    """Online single-attempt judge that issues its own SQUAT/RACK commands.

    Reuses the same signals as training (hip height, thigh scale, per-frame
    depth) plus ``posture.is_locked_out`` and hip-velocity stillness to decide
    when to command. Emits one verdict per attempt with the full fault set:
    INSUFFICIENT_DEPTH, DOWNWARD_MOVEMENT, EARLY_DESCENT (moved before SQUAT),
    EARLY_RACK (left the lockout before RACK) and INCOMPLETE_LOCKOUT.
    """

    def __init__(self, config: CompetitionConfig | None = None) -> None:
        self.config = config or CompetitionConfig()
        self.reset()

    def reset(self) -> None:
        self.state = CompState.AWAIT_SETUP
        self._standing_hip: float | None = None
        self._standing_thigh: float | None = None
        self._motion = MotionTracker(self.config.still_velocity_fraction)
        self._hold = HoldTimer()
        self._lockout_entered: float | None = None
        self._rep_count = 0
        self._last_verdict: RepVerdict | None = None
        self._cand = _Candidate()
        self._early_descent = False
        self._early_rack = False
        self._incomplete_lockout = False
        # Set by _finalize, consumed by the next _status. Keyed to "the attempt was
        # judged", not to a particular command, so every terminal path reports.
        self._just_completed = False
        self._feet = FootDriftMonitor(self.config.posture)

    def update(self, frame: FrameKeypoints3D, depth: DepthFrameResult) -> LiveStatus:
        c = self.config
        hip = mean_height(frame, _HIP, c.min_confidence)
        thigh = segment_length(frame, _THIGH, c.min_confidence)
        below = None if depth.gated else depth.is_below_parallel
        margin = None if depth.gated else depth.depth_margin

        if self.state == CompState.DONE:
            return self._status(below, margin, hip, "attempt complete", scale=self._standing_thigh)
        if hip is None or thigh is None or thigh <= 0:
            self._hold.reset()
            return self._status(below, margin, hip, "waiting for a clear view of hips + knees")

        if self._standing_hip is None:
            self._standing_hip, self._standing_thigh = hip, thigh
        vel = self._motion.observe(hip, frame.time_s)

        scale = self._standing_thigh or thigh
        still = self._motion.is_still(scale)
        locked = is_locked_out(frame, c.posture)  # True / False / None
        t = frame.time_s
        cmd = None
        note = ""

        # The feet must stay planted for the whole attempt, so watch them from the
        # setup hold onward rather than only during the descent.
        if self.state != CompState.AWAIT_SETUP:
            self._feet.observe(frame)

        if self.state == CompState.AWAIT_SETUP:
            if still:
                self._standing_hip = hip
                self._standing_thigh = thigh
            if hip < self._standing_hip - c.enter_fraction * scale and vel < 0:
                self._early_descent = True  # descended before the SQUAT command
                self._begin(frame, hip)
                self.state = CompState.DESCENDING
                note = "moved before the command!"
            elif self._hold.held(t, still and locked is not False, c.setup_hold_s):
                cmd = "SQUAT"
                self.state = CompState.SET
                note = "SQUAT — begin your lift"
            elif still and locked is False:
                note = "stand tall and lock your knees"
            else:
                note = "set up: stand still and locked to get the command"
        elif self.state == CompState.SET:
            if hip < self._standing_hip - c.enter_fraction * scale and vel < 0:
                self._begin(frame, hip)
                self.state = CompState.DESCENDING
                note = "descending…"
            else:
                note = "SQUAT — begin your lift"
        elif self.state == CompState.DESCENDING:
            self._accumulate(depth, hip)
            if hip > self._cand.min_hip + c.bottom_rise_fraction * scale:
                self.state = CompState.ASCENDING
                self._cand.ascent_peak = hip
                note = "stand it up…"
            else:
                note = f"descending… depth {self._frac(hip, scale) * 100:.0f}%"
        elif self.state == CompState.ASCENDING:
            self._accumulate(depth, hip)
            if hip < self._cand.ascent_peak - c.downward_movement_fraction * scale:
                self._cand.downward = True
            self._cand.ascent_peak = max(self._cand.ascent_peak, hip)
            if hip >= self._standing_hip - c.exit_fraction * scale:
                self.state = CompState.AWAIT_LOCKOUT
                self._lockout_entered = t
                self._hold.reset()
                note = "hold it — wait for the rack command"
            else:
                note = "stand it up…"
        elif self.state == CompState.AWAIT_LOCKOUT:
            waited = t - (self._lockout_entered or t)
            if hip < self._standing_hip - c.enter_fraction * scale:
                self._early_rack = True  # broke lockout / re-descended before RACK
                cmd = "RACK"
                note = self._finalize(frame, "left lockout before the rack command!")
            elif self._hold.held(t, still, c.lockout_hold_s) or waited > c.max_lockout_wait_s:
                if locked is False:
                    self._incomplete_lockout = True
                cmd = "RACK"
                note = self._finalize(frame, "RACK")
            else:
                note = "hold it — wait for the rack command"

        return self._status(below, margin, hip, note, command=cmd, scale=scale)

    # -- helpers -------------------------------------------------------------

    def _begin(self, frame: FrameKeypoints3D, hip: float) -> None:
        self._cand = _Candidate(start_frame=frame.frame_idx, start_time=frame.time_s, min_hip=hip)
        self._feet.observe(frame)

    def _accumulate(self, depth: DepthFrameResult, hip: float) -> None:
        self._cand.min_hip = min(self._cand.min_hip, hip)
        if not depth.gated and depth.depth_margin is not None:
            self._cand.had_confident = True
            if depth.is_below_parallel:
                self._cand.reached_below = True
            if self._cand.best_margin is None or depth.depth_margin > self._cand.best_margin:
                self._cand.best_margin = depth.depth_margin
                self._cand.best_conf = depth.confidence

    def _finalize(self, frame: FrameKeypoints3D, note: str) -> str:
        cand = self._cand
        faults: list[Fault] = []
        if self._early_descent:
            faults.append(Fault.EARLY_DESCENT)
        if self._incomplete_lockout:
            faults.append(Fault.INCOMPLETE_LOCKOUT)
        if cand.had_confident and not cand.reached_below:
            faults.append(Fault.INSUFFICIENT_DEPTH)
        if cand.downward:
            faults.append(Fault.DOWNWARD_MOVEMENT)
        if self._feet.moved():
            faults.append(Fault.FOOT_MOVEMENT)
        if self._early_rack:
            faults.append(Fault.EARLY_RACK)

        verdict = decide(faults, uncertain=not cand.had_confident)

        self._last_verdict = RepVerdict(
            rep_index=self._rep_count,
            verdict=verdict,
            confidence=cand.best_conf,
            faults=faults,
            depth_margin=cand.best_margin,
            start_frame=cand.start_frame,
            end_frame=frame.frame_idx,
            start_time_s=cand.start_time,
            end_time_s=frame.time_s,
        )
        self._rep_count += 1
        self.state = CompState.DONE
        self._just_completed = True
        return note

    def _frac(self, hip: float, scale: float) -> float:
        if self._standing_hip is None or scale <= 0:
            return 0.0
        return max(0.0, (self._standing_hip - hip) / scale)

    def _status(
        self,
        below: bool | None,
        margin: float | None,
        hip: float | None,
        note: str,
        *,
        command: str | None = None,
        scale: float | None = None,
    ) -> LiveStatus:
        frac = self._frac(hip, scale) if (hip is not None and scale) else None
        completed, self._just_completed = self._just_completed, False
        return LiveStatus(
            state=self.state,
            note=note,
            below_parallel=below,
            checkpoint=below,  # squat's key checkpoint == below parallel
            depth_margin=margin,
            hip_z=hip,
            standing_ref=self._standing_hip,
            descent_fraction=frac,
            rep_count=self._rep_count,
            last_verdict=self._last_verdict,
            rep_completed=completed,
            command=command,
        )


def lift_frame_to_3d(frame2d: FrameKeypoints, *, fps: float, camera_id: str = "cam0"):
    """Single-view 2D->3D lift for one frame (reuses the fusion fallback)."""
    seq = PoseSequence(camera_id=camera_id, fps=fps, frames=[frame2d])
    return reconstruct_3d([seq]).frames[0]


class LiveJudge:
    """Ties the per-frame pipeline together for a live stream (with smoothing)."""

    def __init__(
        self,
        estimator: PoseEstimator | None = None,
        *,
        fps: float = 30.0,
        depth_config: DepthConfig | None = None,
        tracker: OnlineRepTracker | CompetitionTracker | None = None,
        live_config: LiveConfig | None = None,
    ) -> None:
        self.estimator = estimator or PoseEstimator()
        self.fps = fps
        self.depth_config = depth_config or DepthConfig()
        self.tracker = tracker or OnlineRepTracker(live_config)
        self.smoother = StreamingKeypointSmoother(min_confidence=self._min_conf())
        self._frame_idx = 0

    def _min_conf(self) -> float:
        return getattr(self.tracker.config, "min_confidence", 0.5)

    def set_tracker(self, tracker) -> None:
        """Swap the rep/competition tracker (mode switch), keeping the model warm."""
        self.tracker = tracker
        self.smoother = StreamingKeypointSmoother(min_confidence=self._min_conf())
        self._frame_idx = 0

    def reset(self) -> None:
        """Start fresh: new rep/attempt numbering and a re-learned standing ref."""
        self.tracker.reset()
        self.smoother = StreamingKeypointSmoother(min_confidence=self._min_conf())
        self._frame_idx = 0

    def process_frame(self, bgr_frame) -> tuple[FrameKeypoints, DepthFrameResult, LiveStatus]:
        result = self.estimator.model.predict(
            source=bgr_frame, conf=self.estimator.conf, verbose=False
        )
        raw2d = result_to_frame(result[0], self._frame_idx, self.fps, self.estimator.subject)
        frame2d = self.smoother.smooth(raw2d)
        frame3d = lift_frame_to_3d(frame2d, fps=self.fps)
        depth = judge_depth_frame(frame3d, self.depth_config)
        status = self.tracker.update(frame3d, depth)
        self._frame_idx += 1
        return frame2d, depth, status


# The OpenCV webcam demo now lives in `whitelights.cli` — this module is judging
# logic only. `python -m whitelights.live` is kept working because the README and
# docs have always documented it.
if __name__ == "__main__":  # pragma: no cover
    from .cli import main

    main()
