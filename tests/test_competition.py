"""Tests for the competition (referee-command) online judge.

Drives synthetic attempts frame-by-frame: a still + locked setup (so the judge
issues SQUAT), a descent/ascent, then a still + locked finish (so it issues
RACK), and checks the commands fire and the verdict is right.
"""

from __future__ import annotations

import numpy as np

from whitelights.live import CompetitionTracker
from whitelights.types import Fault, Verdict

STILL = 25  # frames of standing (0.83s at 30fps) — longer than the 0.6s holds


def _ramp(a: float, b: float, n: int = 15) -> list[float]:
    return [float(z) for z in np.linspace(a, b, n)]


def _attempt(bottom: float, *, setup: int = STILL, lockout: int = STILL) -> list[float]:
    return [1.0] * setup + _ramp(1.0, bottom) + _ramp(bottom, 1.0) + [1.0] * lockout


def _drive(tracker, poses, depths):
    return [tracker.update(f, d) for f, d in zip(poses.frames, depths, strict=True)]


def _commands(statuses) -> list[str]:
    return [s.command for s in statuses if s.command]


def test_good_attempt_issues_squat_and_rack(make_full_squat_from_series, make_depth) -> None:
    poses = make_full_squat_from_series(_attempt(0.45))  # breaks parallel
    statuses = _drive(CompetitionTracker(), poses, make_depth(poses))

    assert _commands(statuses) == ["SQUAT", "RACK"]
    final = statuses[-1]
    assert final.rep_count == 1
    assert final.last_verdict.verdict == Verdict.GOOD
    assert final.last_verdict.faults == []


def test_high_attempt_is_no_lift(make_full_squat_from_series, make_depth) -> None:
    poses = make_full_squat_from_series(_attempt(0.60))  # never below parallel
    statuses = _drive(CompetitionTracker(), poses, make_depth(poses))

    assert "SQUAT" in _commands(statuses)
    v = statuses[-1].last_verdict
    assert v.verdict == Verdict.NO_LIFT
    assert Fault.INSUFFICIENT_DEPTH in v.faults


def test_moving_before_squat_flags_early_descent(make_full_squat_from_series, make_depth) -> None:
    # Only 3 setup frames (0.1s) — descend before the 0.6s hold issues SQUAT.
    poses = make_full_squat_from_series(_attempt(0.45, setup=3))
    statuses = _drive(CompetitionTracker(), poses, make_depth(poses))

    assert "SQUAT" not in _commands(statuses)
    v = statuses[-1].last_verdict
    assert Fault.EARLY_DESCENT in v.faults
    assert v.verdict == Verdict.NO_LIFT


def test_leaving_lockout_before_rack_flags_early_rack(
    make_full_squat_from_series, make_depth
) -> None:
    # Good lift, but re-descend at the top before the rack hold completes.
    series = [1.0] * STILL + _ramp(1.0, 0.45) + _ramp(0.45, 1.0) + [1.0] * 3 + _ramp(1.0, 0.5, 8)
    poses = make_full_squat_from_series(series)
    statuses = _drive(CompetitionTracker(), poses, make_depth(poses))

    cmds = _commands(statuses)
    assert "SQUAT" in cmds and "RACK" in cmds
    v = statuses[-1].last_verdict
    assert Fault.EARLY_RACK in v.faults
    assert v.verdict == Verdict.NO_LIFT
