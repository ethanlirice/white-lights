"""Bench press: competition (referee-command) online judge — DRAFT.

Structurally this is the squat competition judge with a different primary signal
and one extra command. Instead of the hip height and hip-below-knee depth, the
bench uses the **bar height (proxied by the wrists)** and a three-command
sandwich: **START** (begin lowering) -> **PRESS** (bar motionless on the chest)
-> **RACK**. Lockout is elbow extension, not knee.

Uncertainty handling (see the accuracy discussion) is built in from the start:
  * per-lifter lockout CALIBRATION — the lifter presents locked arms at setup,
    so we record *their own* elbow lockout angle and grade the finish relative
    to it, not an absolute 180 deg (absorbs anatomy, camera geometry, estimator
    noise);
  * a tolerance band around lockout -> **UNCERTAIN** (too close to call) instead
    of a forced GOOD/NO_LIFT;
  * federation profiles (IPF strict vs USAPL lenient) via ``bench_config_for``.

DRAFT status: the threshold numbers are placeholders to be tuned with labelled
clips via ``eval/validate.py``. Contact points (buttocks/feet), grip, and true
bar detection are deferred (TODO(ethan)). Not yet wired into LiveJudge/WS/UI —
that lands with the lift-selector UI.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum

from pydantic import BaseModel, Field

from .depth import DepthFrameResult
from .live import LiveStatus
from .posture import PostureConfig, elbow_angle_deg
from .types import Fault, FrameKeypoints3D, RepVerdict, Verdict

_WRIST = ("left_wrist", "right_wrist")
_SIDES = ("left", "right")


class Federation(StrEnum):
    IPF = "IPF"  # strict
    USAPL = "USAPL"  # more lenient


class BenchState(StrEnum):
    AWAIT_START = "AWAIT_START"  # arms locked + still at the top -> START
    LOWERING = "LOWERING"  # bar coming down after START
    ON_CHEST = "ON_CHEST"  # motionless on the chest, PRESS given
    PRESSING = "PRESSING"  # bar going up after PRESS
    AWAIT_RACK = "AWAIT_RACK"  # arms locked + still at the top -> RACK
    DONE = "DONE"


class BenchConfig(BaseModel):
    """Thresholds for the bench judge. Distances are fractions of arm length."""

    federation: Federation = Federation.IPF
    min_confidence: float = 0.5
    still_velocity_fraction: float = 0.40
    setup_hold_s: float = 0.60  # still + locked hold before START
    chest_hold_s: float = 0.40  # motionless-on-chest hold before PRESS
    lockout_hold_s: float = 0.60  # still hold at the top before RACK
    lower_enter_fraction: float = 0.15  # bar drop that counts as "lowering"
    min_touch_fraction: float = 0.55  # descent from the top that counts as to-chest
    rise_fraction: float = 0.05  # bar rising past the bottom -> pressing
    exit_fraction: float = 0.15  # bar back near the extended top
    downward_movement_fraction: float = 0.05  # re-descent during the press -> fault
    # Lockout graded vs the calibrated setup elbow angle: within `tolerance` is
    # locked; a further `uncertain` band beyond that is too-close-to-call.
    lockout_tolerance_deg: float = 8.0
    lockout_uncertain_deg: float = 6.0
    max_wait_s: float = 8.0
    posture: PostureConfig = Field(default_factory=PostureConfig)


def bench_config_for(federation: Federation) -> BenchConfig:
    """Federation strictness profile (thresholds + UNCERTAIN band width)."""
    if federation == Federation.USAPL:
        return BenchConfig(
            federation=Federation.USAPL,
            min_touch_fraction=0.50,
            lockout_tolerance_deg=12.0,
            lockout_uncertain_deg=8.0,
        )
    return BenchConfig(federation=Federation.IPF)


@dataclass
class _Bench:
    start_frame: int = 0
    start_time: float = 0.0
    min_wrist: float = math.inf
    press_peak: float = -math.inf
    reached_chest: bool = False
    downward: bool = False


class BenchTracker:
    """Online single-attempt bench judge that issues its own START/PRESS/RACK."""

    def __init__(self, config: BenchConfig | None = None) -> None:
        self.config = config or BenchConfig()
        self.reset()

    def reset(self) -> None:
        self.state = BenchState.AWAIT_START
        self._top_z: float | None = None  # extended-arms wrist height (calibrated)
        self._arm_len: float | None = None
        self._lock_ref: float | None = None  # per-lifter elbow lockout angle (calibrated)
        self._prev: float | None = None
        self._prev_t: float | None = None
        self._hold: float | None = None
        self._lockout_entered: float | None = None
        self._rep_count = 0
        self._last_verdict: RepVerdict | None = None
        self._cand = _Bench()
        self._early_descent = False
        self._early_press = False
        self._early_rack = False
        self._incomplete_lockout = False
        self._lockout_uncertain = False

    def update(self, frame: FrameKeypoints3D, depth: DepthFrameResult) -> LiveStatus:
        c = self.config
        wrist = _mean_z(frame, _WRIST, c.min_confidence)
        arm = _arm_len(frame, c.min_confidence)
        elbow = _elbow_angle(frame, c.min_confidence)

        if self.state == BenchState.DONE:
            return self._status(wrist, "attempt complete")
        if wrist is None or arm is None or arm <= 0:
            self._hold = None
            return self._status(wrist, "waiting for a clear view of your arms")

        if self._top_z is None:
            self._top_z, self._arm_len = wrist, arm
        dt = frame.time_s - self._prev_t if self._prev_t is not None else None
        vel = (wrist - self._prev) / dt if (dt and dt > 0 and self._prev is not None) else 0.0
        self._prev, self._prev_t = wrist, frame.time_s

        scale = self._arm_len or arm
        still = abs(vel) < c.still_velocity_fraction * scale
        t = frame.time_s
        locked_setup = elbow is not None and elbow >= c.posture.lockout_elbow_angle_deg
        cmd = None
        note = ""

        if self.state == BenchState.AWAIT_START:
            if still and locked_setup:  # calibrate the lifter's own top + lockout
                self._top_z, self._arm_len, self._lock_ref = wrist, arm, elbow
            if wrist < self._top_z - c.lower_enter_fraction * scale and vel < 0:
                self._early_descent = True
                self._begin(frame, wrist)
                self.state = BenchState.LOWERING
                note = "lowered before the command!"
            elif self._held(t, still and locked_setup, c.setup_hold_s):
                cmd = "START"
                self._begin(frame, wrist)
                self.state = BenchState.LOWERING
                note = "START — lower the bar to your chest"
            elif still and elbow is not None and not locked_setup:
                note = "lock your arms out to get the start command"
            else:
                note = "hold the bar still and locked to start"
        elif self.state == BenchState.LOWERING:
            self._cand.min_wrist = min(self._cand.min_wrist, wrist)
            descended = self._top_z - self._cand.min_wrist
            if descended >= c.min_touch_fraction * scale:
                self._cand.reached_chest = True
            if wrist > self._cand.min_wrist + c.rise_fraction * scale and vel > 0:
                self._early_press = True  # pressed before the PRESS command
                self._cand.press_peak = wrist
                self.state = BenchState.PRESSING
                note = "pressed before the command!"
            elif self._held(t, still and self._cand.reached_chest, c.chest_hold_s):
                cmd = "PRESS"
                self.state = BenchState.ON_CHEST
                note = "PRESS — press it up"
            else:
                note = (
                    f"lowering… {min(1.0, descended / (c.min_touch_fraction * scale)) * 100:.0f}%"
                )
        elif self.state == BenchState.ON_CHEST:
            if wrist > self._cand.min_wrist + c.rise_fraction * scale and vel > 0:
                self._cand.press_peak = wrist
                self.state = BenchState.PRESSING
                note = "pressing…"
            else:
                note = "PRESS — press it up"
        elif self.state == BenchState.PRESSING:
            if wrist < self._cand.press_peak - c.downward_movement_fraction * scale:
                self._cand.downward = True
            self._cand.press_peak = max(self._cand.press_peak, wrist)
            if wrist >= self._top_z - c.exit_fraction * scale:
                self.state = BenchState.AWAIT_RACK
                self._lockout_entered = t
                self._hold = None
                note = "hold it — wait for the rack command"
            else:
                note = "pressing…"
        elif self.state == BenchState.AWAIT_RACK:
            waited = t - (self._lockout_entered or t)
            if wrist < self._top_z - c.lower_enter_fraction * scale:
                self._early_rack = True
                cmd = "RACK"
                note = self._finalize(frame, elbow, "racked before the command!")
            elif self._held(t, still, c.lockout_hold_s) or waited > c.max_wait_s:
                cmd = "RACK"
                note = self._finalize(frame, elbow, "RACK")
            else:
                note = "hold it — wait for the rack command"

        return self._status(wrist, note, command=cmd, scale=scale)

    # -- helpers -------------------------------------------------------------

    def _held(self, t: float, condition: bool, hold_s: float) -> bool:
        if not condition:
            self._hold = None
            return False
        if self._hold is None:
            self._hold = t
        return (t - self._hold) >= hold_s

    def _begin(self, frame: FrameKeypoints3D, wrist: float) -> None:
        self._cand = _Bench(start_frame=frame.frame_idx, start_time=frame.time_s, min_wrist=wrist)

    def _grade_lockout(self, elbow: float | None) -> None:
        c = self.config
        if elbow is None or self._lock_ref is None:
            self._lockout_uncertain = True
            return
        deficit = self._lock_ref - elbow
        if deficit <= c.lockout_tolerance_deg:
            return  # locked
        if deficit <= c.lockout_tolerance_deg + c.lockout_uncertain_deg:
            self._lockout_uncertain = True  # too close to call
        else:
            self._incomplete_lockout = True

    def _finalize(self, frame: FrameKeypoints3D, elbow: float | None, note: str) -> str:
        self._grade_lockout(elbow)
        faults: list[Fault] = []
        if self._early_descent:
            faults.append(Fault.EARLY_DESCENT)
        if self._early_press:
            faults.append(Fault.EARLY_PRESS)
        if not self._cand.reached_chest:
            faults.append(Fault.BAR_NOT_TO_CHEST)
        if self._incomplete_lockout:
            faults.append(Fault.INCOMPLETE_LOCKOUT)
        if self._cand.downward:
            faults.append(Fault.DOWNWARD_MOVEMENT)
        if self._early_rack:
            faults.append(Fault.EARLY_RACK)

        if faults:
            verdict = Verdict.NO_LIFT
        elif self._lockout_uncertain:
            verdict = Verdict.UNCERTAIN  # too close to call — don't fake a call
        else:
            verdict = Verdict.GOOD

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
        self.state = BenchState.DONE
        return note

    def _status(
        self,
        wrist: float | None,
        note: str,
        *,
        command: str | None = None,
        scale: float | None = None,
    ) -> LiveStatus:
        frac = None
        if wrist is not None and self._top_z is not None and scale:
            reach = self.config.min_touch_fraction * scale
            frac = max(0.0, min(1.0, (self._top_z - wrist) / reach)) if reach else None
        return LiveStatus(
            state=self.state,
            note=note,
            below_parallel=None,
            checkpoint=self._cand.reached_chest,  # bench checkpoint: bar reached the chest
            depth_margin=None,
            hip_z=wrist,
            standing_ref=self._top_z,
            descent_fraction=frac,
            rep_count=self._rep_count,
            last_verdict=self._last_verdict,
            rep_completed=(self.state == BenchState.DONE and command == "RACK"),
            command=command,
        )


def _mean_z(frame: FrameKeypoints3D, names: tuple[str, ...], min_conf: float) -> float | None:
    zs = [kp.z for n in names if (kp := frame.get(n)) is not None and kp.confidence >= min_conf]
    return sum(zs) / len(zs) if zs else None


def _arm_len(frame: FrameKeypoints3D, min_conf: float) -> float | None:
    """Shoulder->elbow->wrist segment length (constant scale, unlike reach)."""
    lengths: list[float] = []
    for side in _SIDES:
        s = frame.get(f"{side}_shoulder")
        e = frame.get(f"{side}_elbow")
        w = frame.get(f"{side}_wrist")
        if s is None or e is None or w is None:
            continue
        if min(s.confidence, e.confidence, w.confidence) < min_conf:
            continue
        lengths.append(
            math.dist((s.x, s.y, s.z), (e.x, e.y, e.z))
            + math.dist((e.x, e.y, e.z), (w.x, w.y, w.z))
        )
    return sum(lengths) / len(lengths) if lengths else None


def _elbow_angle(frame: FrameKeypoints3D, min_conf: float) -> float | None:
    """The more-bent elbow angle (governs lockout), or None if unmeasurable."""
    angles = [a for side in _SIDES if (a := elbow_angle_deg(frame, side, min_conf)) is not None]
    return min(angles) if angles else None
