"""Deadlift: competition (referee-command) online judge — DRAFT.

The deadlift has a single command — **DOWN**, given at the top. The lifter pulls
the bar from the floor to a standing lockout (knees locked **and** hips/torso
extended, upright), holds motionless, and only lowers on the command.

State machine::

    AWAIT_LIFT -> PULLING -> AWAIT_DOWN -> DONE
                     |
                     +-- gave up / re-descended without locking -> INCOMPLETE_LOCKOUT

Primary signal is the **bar height (proxied by the wrists)**; lockout is a
posture condition — knee angle (hip-knee-ankle) *and* hip/torso angle
(shoulder-hip-knee) both near-straight. Faults: DOWNWARD_MOVEMENT (any downward
bar movement during the pull — the deadlift is strict here), INCOMPLETE_LOCKOUT,
EARLY_DOWN (lowered before the command).

Uncertainty handling: lockout is graded against configurable knee/hip thresholds
with a tolerance band → **UNCERTAIN** (too close to call) rather than a forced
call. Federation strictness via ``deadlift_config_for``.

DRAFT: numbers are placeholders to tune with labelled clips. HITCHING (ramping
the bar up the thighs) and BAR_SUPPORTED_ON_THIGHS are deferred (TODO(ethan)).
Not yet wired into LiveJudge/WS/UI — lands with the lift selector.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum

from pydantic import BaseModel, Field

from .depth import DepthFrameResult
from .federations import Federation
from .posture import FootDriftMonitor, PostureConfig
from .tracking import (
    HoldTimer,
    MotionTracker,
    decide,
    governing_angle,
    mean_height,
    segment_length,
)
from .types import Fault, FrameKeypoints3D, LiveStatus, RepVerdict, Verdict

_WRIST = ("left_wrist", "right_wrist")
_TORSO = ("shoulder", "hip")
_KNEE_CHAIN = ("hip", "knee", "ankle")
_HIP_CHAIN = ("shoulder", "hip", "knee")


class DeadliftState(StrEnum):
    AWAIT_LIFT = "AWAIT_LIFT"  # bar on the floor, waiting for the pull
    PULLING = "PULLING"  # bar rising toward lockout
    AWAIT_DOWN = "AWAIT_DOWN"  # locked out, waiting for the DOWN command
    DONE = "DONE"


class DeadliftConfig(BaseModel):
    """Thresholds for the deadlift judge. Distances are fractions of torso length
    (shoulder-hip) — a body-scale reference that stays stable through the pull,
    unlike the thigh, which foreshortens as the knee bends."""

    federation: Federation = Federation.IPF
    min_confidence: float = 0.5
    still_velocity_fraction: float = 0.40
    down_hold_s: float = 0.60  # still hold at lockout before DOWN
    lift_enter_fraction: float = 0.25  # bar rise off the floor that starts the pull
    downward_movement_fraction: float = 0.04  # any re-descent past this -> fault (strict)
    abort_fraction: float = 0.35  # big re-descent without lockout -> failed lift
    lockout_knee_angle_deg: float = 165.0
    lockout_hip_angle_deg: float = 160.0
    lockout_uncertain_deg: float = 6.0  # band above threshold that reads as too-close-to-call
    max_wait_s: float = 8.0
    posture: PostureConfig = Field(default_factory=PostureConfig)


def deadlift_config_for(federation: Federation) -> DeadliftConfig:
    """Federation strictness profile."""
    if federation == Federation.USAPL:
        return DeadliftConfig(
            federation=Federation.USAPL,
            lockout_knee_angle_deg=160.0,
            lockout_hip_angle_deg=155.0,
            lockout_uncertain_deg=8.0,
        )
    return DeadliftConfig(federation=Federation.IPF)


@dataclass
class _DL:
    start_frame: int = 0
    start_time: float = 0.0
    peak_bar: float = -math.inf
    lockout_bar: float = 0.0
    downward: bool = False


class DeadliftTracker:
    """Online single-attempt deadlift judge that issues its own DOWN command."""

    judges_depth = False  # see `OnlineRepTracker.judges_depth`

    def __init__(self, config: DeadliftConfig | None = None) -> None:
        self.config = config or DeadliftConfig()
        self.reset()

    def reset(self) -> None:
        self.state = DeadliftState.AWAIT_LIFT
        self._floor: float | None = None
        self._torso: float | None = None
        self._motion = MotionTracker(self.config.still_velocity_fraction)
        self._hold = HoldTimer()
        self._lockout_entered: float | None = None
        self._rep_count = 0
        self._last_verdict: RepVerdict | None = None
        self._cand = _DL()
        self._incomplete = False
        self._early_down = False
        self._uncertain = False
        # Set by _finalize, consumed by the next _status. Completion is a property
        # of having judged the attempt, not of which command happened to be issued
        # — the abort path (never locked out) finalises without a command.
        self._just_completed = False
        self._feet = FootDriftMonitor(self.config.posture)

    def update(self, frame: FrameKeypoints3D, depth: DepthFrameResult) -> LiveStatus:
        c = self.config
        bar = mean_height(frame, _WRIST, c.min_confidence)
        torso = segment_length(frame, _TORSO, c.min_confidence)
        knee = governing_angle(frame, _KNEE_CHAIN, c.min_confidence)
        hip = governing_angle(frame, _HIP_CHAIN, c.min_confidence)

        if self.state == DeadliftState.DONE:
            return self._status(bar, "attempt complete")
        if bar is None or torso is None or torso <= 0:
            self._hold.reset()
            return self._status(bar, "waiting for a clear view of the bar + legs")

        if self._floor is None:
            self._floor, self._torso = bar, torso
        self._motion.observe(bar, frame.time_s)

        scale = self._torso or torso
        still = self._motion.is_still(scale)
        locked = (
            knee is not None
            and hip is not None
            and knee >= c.lockout_knee_angle_deg
            and hip >= c.lockout_hip_angle_deg
        )
        t = frame.time_s
        cmd = None
        note = ""

        if self.state == DeadliftState.AWAIT_LIFT:
            self._floor = min(self._floor, bar)
            if bar > self._floor + c.lift_enter_fraction * scale:
                self._begin(frame, bar)
                self.state = DeadliftState.PULLING
                note = "pulling…"
            else:
                note = "set up and pull the bar when ready"
        elif self.state == DeadliftState.PULLING:
            self._feet.observe(frame)
            if bar < self._cand.peak_bar - c.downward_movement_fraction * scale:
                self._cand.downward = True
            if bar < self._cand.peak_bar - c.abort_fraction * scale and not locked:
                self._incomplete = True  # came back down without locking out
                note = self._finalize(frame, knee, hip, "no lift — never locked out")
            elif locked and still:
                self.state = DeadliftState.AWAIT_DOWN
                self._cand.lockout_bar = bar
                self._lockout_entered = t
                self._hold.reset()
                note = "locked — hold for the down command"
            else:
                self._cand.peak_bar = max(self._cand.peak_bar, bar)
                note = "pull to a full lockout — knees and hips straight"
        elif self.state == DeadliftState.AWAIT_DOWN:
            self._feet.observe(frame)
            waited = t - (self._lockout_entered or t)
            if bar < self._cand.lockout_bar - c.downward_movement_fraction * 2 * scale:
                self._early_down = True
                cmd = "DOWN"
                note = self._finalize(frame, knee, hip, "lowered before the down command!")
            elif self._hold.held(t, still, c.down_hold_s) or waited > c.max_wait_s:
                cmd = "DOWN"
                note = self._finalize(frame, knee, hip, "DOWN")
            else:
                note = "hold it — wait for the down command"

        return self._status(bar, note, command=cmd, scale=scale, checkpoint=locked)

    # -- helpers -------------------------------------------------------------

    def _begin(self, frame: FrameKeypoints3D, bar: float) -> None:
        self._cand = _DL(start_frame=frame.frame_idx, start_time=frame.time_s, peak_bar=bar)
        self._feet.reset()
        self._feet.observe(frame)

    def _grade_lockout(self, knee: float | None, hip: float | None) -> None:
        c = self.config
        if knee is None or hip is None:
            self._uncertain = True
            return
        if knee < c.lockout_knee_angle_deg + c.lockout_uncertain_deg or (
            hip < c.lockout_hip_angle_deg + c.lockout_uncertain_deg
        ):
            self._uncertain = True  # only just locked -> too close to call

    def _finalize(
        self, frame: FrameKeypoints3D, knee: float | None, hip: float | None, note: str
    ) -> str:
        if not self._incomplete and not self._early_down:
            self._grade_lockout(knee, hip)
        faults: list[Fault] = []
        if self._incomplete:
            faults.append(Fault.INCOMPLETE_LOCKOUT)
        if self._cand.downward:
            faults.append(Fault.DOWNWARD_MOVEMENT)
        if self._feet.moved():
            faults.append(Fault.FOOT_MOVEMENT)
        if self._early_down:
            faults.append(Fault.EARLY_DOWN)

        verdict = decide(faults, uncertain=self._uncertain)

        self._last_verdict = RepVerdict(
            rep_index=self._rep_count,
            verdict=verdict,
            confidence=0.5 if verdict == Verdict.UNCERTAIN else 0.9,
            faults=faults,
            depth_margin=None,
            start_frame=self._cand.start_frame,
            end_frame=frame.frame_idx,
            start_time_s=self._cand.start_time,
            end_time_s=frame.time_s,
        )
        self._rep_count += 1
        self.state = DeadliftState.DONE
        self._just_completed = True
        return note

    def _status(
        self,
        bar: float | None,
        note: str,
        *,
        command: str | None = None,
        scale: float | None = None,
        checkpoint: bool | None = None,
    ) -> LiveStatus:
        frac = None
        if bar is not None and self._floor is not None and scale:
            frac = max(0.0, min(1.0, (bar - self._floor) / scale))
        completed, self._just_completed = self._just_completed, False
        return LiveStatus(
            state=self.state,
            note=note,
            below_parallel=None,
            checkpoint=checkpoint,  # deadlift checkpoint: locked out (knees + hips straight)
            depth_margin=None,
            hip_z=bar,
            standing_ref=self._floor,
            descent_fraction=frac,
            rep_count=self._rep_count,
            last_verdict=self._last_verdict,
            rep_completed=completed,
            command=command,
        )
