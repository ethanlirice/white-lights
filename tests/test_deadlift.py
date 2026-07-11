"""Tests for the competition deadlift judge (DeadliftTracker).

Drives synthetic attempts frame-by-frame: the bar is pulled from the floor to a
standing lockout, held, then (on a good lift) the DOWN command is issued.
Deadlift ignores the depth arg, so a gated placeholder is passed per frame.
"""

from __future__ import annotations

import numpy as np

from whitelights.deadlift import DeadliftTracker
from whitelights.depth import DepthFrameResult
from whitelights.types import Fault, Verdict

STILL = 25
FLOOR = 0.15
TOP = 1.0


def _ramp(a: float, b: float, n: int = 20) -> list[float]:
    return [float(z) for z in np.linspace(a, b, n)]


def _drive(tracker, poses):
    depths = [
        DepthFrameResult(frame_idx=f.frame_idx, time_s=f.time_s, gated=True) for f in poses.frames
    ]
    return [tracker.update(f, d) for f, d in zip(poses.frames, depths, strict=True)]


def _commands(statuses):
    return [s.command for s in statuses if s.command]


def test_good_deadlift_issues_down(make_deadlift_from_series) -> None:
    series = [FLOOR] * STILL + _ramp(FLOOR, TOP) + [TOP] * STILL + _ramp(TOP, FLOOR, 15)
    statuses = _drive(DeadliftTracker(), make_deadlift_from_series(series))

    assert _commands(statuses) == ["DOWN"]
    v = statuses[-1].last_verdict
    assert v.verdict == Verdict.GOOD
    assert v.faults == []


def test_downward_movement_during_pull(make_deadlift_from_series) -> None:
    series = (
        [FLOOR] * STILL
        + _ramp(FLOOR, 0.6, 12)
        + _ramp(0.6, 0.45, 6)  # the bar dips on the way up
        + _ramp(0.45, TOP, 12)
        + [TOP] * STILL
    )
    statuses = _drive(DeadliftTracker(), make_deadlift_from_series(series))

    v = statuses[-1].last_verdict
    assert Fault.DOWNWARD_MOVEMENT in v.faults
    assert v.verdict == Verdict.NO_LIFT


def test_lowering_before_down_command(make_deadlift_from_series) -> None:
    # Reach lockout, then lower before the down hold completes.
    series = [FLOOR] * STILL + _ramp(FLOOR, TOP) + [TOP] * 3 + _ramp(TOP, FLOOR, 15)
    statuses = _drive(DeadliftTracker(), make_deadlift_from_series(series))

    v = statuses[-1].last_verdict
    assert Fault.EARLY_DOWN in v.faults
    assert v.verdict == Verdict.NO_LIFT


def test_never_locking_out_is_incomplete(make_deadlift_from_series) -> None:
    # Pull only partway (never a full lockout), hold, then set it down.
    series = [FLOOR] * STILL + _ramp(FLOOR, 0.7) + [0.7] * 10 + _ramp(0.7, FLOOR, 15)
    statuses = _drive(DeadliftTracker(), make_deadlift_from_series(series))

    assert "DOWN" not in _commands(statuses)  # never earned the command
    v = statuses[-1].last_verdict
    assert Fault.INCOMPLETE_LOCKOUT in v.faults
    assert v.verdict == Verdict.NO_LIFT
