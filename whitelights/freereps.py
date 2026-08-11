"""Free-rep (training) judges for the bench press and deadlift.

Training mode counts reps and calls each one; nobody issues commands and the set
runs until the lifter stops. Until now only the squat had such a judge, so
selecting training for bench or deadlift silently handed back the single-attempt
competition judge — a different interaction wearing the wrong label.

One tracker serves both lifts. Unlike the command-sandwich judges — whose state
graphs genuinely differ, which is why `tracking.py` deliberately does not unify
them — every free-rep cycle is the same shape: leave a resting position, reach an
extreme, come back. The lifts differ only in which direction that is:

    bench     wrist height, arms extended -> chest -> extended     (out = down)
    deadlift  bar height,   floor -> standing lockout -> floor     (out = up)

Both collapse to one code path by working in **travel from rest**, defined as
``direction * (signal - rest)``, which increases on the way out and returns to
zero at the end whichever way the lifter actually moves.

Judging is deliberately conservative, matching the squat tracker: the resting
reference re-baselines only while the lifter is still, a rep must clear a minimum
travel *and* a minimum duration before it counts (so a shrug or a single noisy
frame is discarded rather than scored), and a checkpoint that could not be
measured yields UNCERTAIN instead of a guess.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum

from pydantic import BaseModel, Field

from .depth import DepthFrameResult
from .federations import Federation
from .posture import FootDriftMonitor, PostureConfig
from .tracking import HoldTimer, MotionTracker, decide, governing_angle, mean_height, segment_length
from .types import Fault, FrameKeypoints3D, LiveStatus, RepVerdict

_WRIST = ("left_wrist", "right_wrist")
_ARM = ("shoulder", "elbow", "wrist")
_TORSO = ("shoulder", "hip")
_KNEE_CHAIN = ("hip", "knee", "ankle")
_HIP_CHAIN = ("shoulder", "hip", "knee")

DESCEND_FIRST = -1  # signal falls on the way out (bench)
ASCEND_FIRST = +1  # signal rises on the way out (deadlift)


class FreeRepState(StrEnum):
    """Names are shown verbatim in the UI; the moving ones pulse the lamp."""

    READY = "READY"
    LOWERING = "LOWERING"
    PRESSING = "PRESSING"
    PULLING = "PULLING"


class FreeRepConfig(BaseModel):
    """Thresholds for a free-rep set. Distances are fractions of body scale."""

    min_confidence: float = 0.5
    enter_fraction: float = 0.20  # travel from rest that starts a rep
    exit_fraction: float = 0.12  # travel below which the lifter is back at rest
    turn_fraction: float = 0.05  # pull-back from the extreme that marks the turnaround
    reversal_fraction: float = 0.05  # travel regained on the way back -> DOWNWARD_MOVEMENT
    min_rep_travel_fraction: float = 0.45  # total travel required to count as a rep
    min_rep_duration_s: float = 0.40  # rejects flickers
    still_velocity_fraction: float = 0.45
    rest_ema: float = 0.20  # how fast the resting reference re-baselines when still
    max_lost_frames: int = 8
    posture: PostureConfig = Field(default_factory=PostureConfig)


# ---------------------------------------------------------------------------
# Per-lift specification
# ---------------------------------------------------------------------------

# A checkpoint answers "did this rep reach the position the rulebook requires?"
# for the current frame: True / False / None when it cannot be measured.
Checkpoint = Callable[[FrameKeypoints3D, DepthFrameResult, float, float], bool | None]


@dataclass(frozen=True)
class FreeRepSpec:
    """What distinguishes one lift's free reps from another's."""

    lift: str
    signal: tuple[str, ...]  # keypoints averaged into the primary signal
    scale_chain: tuple[str, ...]  # body-scale reference
    direction: int  # DESCEND_FIRST or ASCEND_FIRST
    checkpoint: Checkpoint
    checkpoint_fault: Fault  # raised when the checkpoint was measurable but never met
    outbound_state: FreeRepState
    return_state: FreeRepState
    outbound_note: str
    return_note: str
    checkpoint_hint: str  # coaching line while the checkpoint is unmet
    watch_feet: bool = False


def bench_free_rep_spec(config: FreeRepConfig, *, touch_fraction: float = 0.55) -> FreeRepSpec:
    """Bench: the bar must reach the chest, measured as wrist travel from lockout."""

    def reached_chest(
        frame: FrameKeypoints3D, depth: DepthFrameResult, travel: float, scale: float
    ) -> bool | None:
        if scale <= 0:
            return None
        return travel >= touch_fraction * scale

    return FreeRepSpec(
        lift="bench",
        signal=_WRIST,
        scale_chain=_ARM,
        direction=DESCEND_FIRST,
        checkpoint=reached_chest,
        checkpoint_fault=Fault.BAR_NOT_TO_CHEST,
        outbound_state=FreeRepState.LOWERING,
        return_state=FreeRepState.PRESSING,
        outbound_note="lowering to the chest…",
        return_note="pressing…",
        checkpoint_hint="bring the bar all the way to your chest",
    )


def deadlift_free_rep_spec(config: FreeRepConfig) -> FreeRepSpec:
    """Deadlift: the lifter must reach a standing lockout — knees *and* hips straight."""
    knee_min = 165.0
    hip_min = 160.0

    def locked_out(
        frame: FrameKeypoints3D, depth: DepthFrameResult, travel: float, scale: float
    ) -> bool | None:
        knee = governing_angle(frame, _KNEE_CHAIN, config.min_confidence)
        hip = governing_angle(frame, _HIP_CHAIN, config.min_confidence)
        if knee is None or hip is None:
            return None
        return knee >= knee_min and hip >= hip_min

    return FreeRepSpec(
        lift="deadlift",
        signal=_WRIST,
        scale_chain=_TORSO,
        direction=ASCEND_FIRST,
        checkpoint=locked_out,
        checkpoint_fault=Fault.INCOMPLETE_LOCKOUT,
        outbound_state=FreeRepState.PULLING,
        return_state=FreeRepState.LOWERING,
        outbound_note="pulling…",
        return_note="lowering under control…",
        checkpoint_hint="stand it up — knees and hips straight",
        watch_feet=True,
    )


@dataclass
class _Rep:
    """Evidence for a rep in progress, all in travel-from-rest space."""

    start_frame: int = 0
    start_time: float = 0.0
    peak_travel: float = 0.0
    return_low: float = math.inf  # least travel seen on the way back
    checkpoint_met: bool = False
    checkpoint_seen: bool = False  # was it ever measurable?
    reversed_direction: bool = False
    lost: int = 0
    faults: list[Fault] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Tracker
# ---------------------------------------------------------------------------


class FreeRepTracker:
    """Counts and judges free reps for any out-and-back lift.

    Same interface as every other online judge — ``update(frame, depth)`` and
    ``reset()`` — so `LiveJudge` and the WebSocket handler stay lift-agnostic.
    """

    # The free-rep specs that exist are bench and deadlift; squat free reps use
    # `OnlineRepTracker`. See `OnlineRepTracker.judges_depth`.
    judges_depth = False

    def __init__(self, spec: FreeRepSpec, config: FreeRepConfig | None = None) -> None:
        self.spec = spec
        self.config = config or FreeRepConfig()
        self.reset()

    def reset(self) -> None:
        self.state = FreeRepState.READY
        self._rest: float | None = None
        self._scale: float | None = None
        self._motion = MotionTracker(self.config.still_velocity_fraction)
        self._hold = HoldTimer()
        self._rep_count = 0
        self._last_verdict: RepVerdict | None = None
        self._rep = _Rep()
        self._feet = FootDriftMonitor(self.config.posture)
        self._just_completed = False

    # -- main loop -----------------------------------------------------------

    def update(self, frame: FrameKeypoints3D, depth: DepthFrameResult) -> LiveStatus:
        c = self.config
        spec = self.spec
        signal = mean_height(frame, spec.signal, c.min_confidence)
        scale_now = segment_length(frame, spec.scale_chain, c.min_confidence)

        if signal is None or scale_now is None or scale_now <= 0:
            if self.state != FreeRepState.READY:
                self._rep.lost += 1
                if self._rep.lost > c.max_lost_frames:
                    self.state = FreeRepState.READY  # abandon the rep, don't score it
                    return self._status(None, None, "lost you — start the rep again")
            return self._status(None, None, "waiting for a clear view")

        if self._rest is None:
            self._rest, self._scale = signal, scale_now
        self._motion.observe(signal, frame.time_s)
        scale = self._scale or scale_now
        still = self._motion.is_still(scale)

        travel = self._travel(signal)
        checkpoint = spec.checkpoint(frame, depth, travel, scale)
        note = ""

        if self.state == FreeRepState.READY:
            if still:  # re-baseline rest only while genuinely still
                a = c.rest_ema
                self._rest = (1 - a) * self._rest + a * signal
                self._scale = (1 - a) * (self._scale or scale_now) + a * scale_now
                scale = self._scale
                travel = self._travel(signal)
            if travel > c.enter_fraction * scale:
                self._begin(frame)
                self.state = spec.outbound_state
                note = spec.outbound_note
            else:
                plural = "" if self._rep_count == 1 else "s"
                note = f"ready — {self._rep_count} rep{plural} this set"
        elif self.state == spec.outbound_state:
            self._accumulate(frame, checkpoint, travel)
            if travel < self._rep.peak_travel - c.turn_fraction * scale:
                self.state = spec.return_state
                self._rep.return_low = travel
                note = spec.return_note
            elif checkpoint is False:
                note = spec.checkpoint_hint
            else:
                note = spec.outbound_note
        elif self.state == spec.return_state:
            self._accumulate(frame, checkpoint, travel)
            if travel > self._rep.return_low + c.reversal_fraction * scale:
                self._rep.reversed_direction = True
            self._rep.return_low = min(self._rep.return_low, travel)
            if travel <= c.exit_fraction * scale:
                note = self._complete(frame, scale)
                self.state = FreeRepState.READY
                self._rest = signal
            else:
                note = spec.return_note

        return self._status(travel, checkpoint, note, scale=scale)

    # -- rep lifecycle -------------------------------------------------------

    def _travel(self, signal: float) -> float:
        """Distance moved away from rest, positive whichever way the lift goes."""
        if self._rest is None:
            return 0.0
        return self.spec.direction * (signal - self._rest)

    def _begin(self, frame: FrameKeypoints3D) -> None:
        self._rep = _Rep(start_frame=frame.frame_idx, start_time=frame.time_s)
        if self.spec.watch_feet:
            self._feet.reset()
            self._feet.observe(frame)

    def _accumulate(self, frame: FrameKeypoints3D, checkpoint: bool | None, travel: float) -> None:
        self._rep.lost = 0
        self._rep.peak_travel = max(self._rep.peak_travel, travel)
        if checkpoint is not None:
            self._rep.checkpoint_seen = True
            if checkpoint:
                self._rep.checkpoint_met = True
        if self.spec.watch_feet:
            self._feet.observe(frame)

    def _complete(self, frame: FrameKeypoints3D, scale: float) -> str:
        c = self.config
        rep = self._rep
        duration = frame.time_s - rep.start_time
        if rep.peak_travel < c.min_rep_travel_fraction * scale or duration < c.min_rep_duration_s:
            return "partial movement — not counted"

        faults: list[Fault] = []
        if rep.checkpoint_seen and not rep.checkpoint_met:
            faults.append(self.spec.checkpoint_fault)
        if rep.reversed_direction:
            faults.append(Fault.DOWNWARD_MOVEMENT)
        if self.spec.watch_feet and self._feet.moved():
            faults.append(Fault.FOOT_MOVEMENT)

        verdict = decide(faults, uncertain=not rep.checkpoint_seen)
        self._last_verdict = RepVerdict(
            rep_index=self._rep_count,
            verdict=verdict,
            confidence=0.5 if not rep.checkpoint_seen else 0.9,
            faults=faults,
            depth_margin=None,
            start_frame=rep.start_frame,
            end_frame=frame.frame_idx,
            start_time_s=rep.start_time,
            end_time_s=frame.time_s,
        )
        self._rep_count += 1
        self._just_completed = True
        label = verdict.value + (f" — {', '.join(f.value for f in faults)}" if faults else "")
        return f"REP {self._rep_count}: {label}"

    def _status(
        self,
        travel: float | None,
        checkpoint: bool | None,
        note: str,
        *,
        scale: float | None = None,
    ) -> LiveStatus:
        progress = None
        if travel is not None and scale:
            target = self.config.min_rep_travel_fraction * scale
            progress = max(0.0, min(1.0, travel / target)) if target else None
        completed, self._just_completed = self._just_completed, False
        return LiveStatus(
            state=self.state,
            note=note,
            below_parallel=None,
            checkpoint=checkpoint,
            depth_margin=None,
            hip_z=travel,
            standing_ref=self._rest,
            descent_fraction=progress,
            rep_count=self._rep_count,
            last_verdict=self._last_verdict,
            rep_completed=completed,
            command=None,  # free reps are uncommanded, by definition
        )


# ---------------------------------------------------------------------------
# Constructors
# ---------------------------------------------------------------------------


def bench_rep_tracker(
    config: FreeRepConfig | None = None, *, federation: Federation = Federation.IPF
) -> FreeRepTracker:
    """Free-rep bench judge. USAPL accepts a slightly shallower chest touch."""
    config = config or FreeRepConfig()
    touch = 0.50 if federation == Federation.USAPL else 0.55
    return FreeRepTracker(bench_free_rep_spec(config, touch_fraction=touch), config)


def deadlift_rep_tracker(config: FreeRepConfig | None = None) -> FreeRepTracker:
    """Free-rep deadlift judge."""
    config = config or FreeRepConfig()
    return FreeRepTracker(deadlift_free_rep_spec(config), config)
