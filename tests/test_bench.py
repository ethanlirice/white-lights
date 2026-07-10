"""Tests for the competition bench-press judge (BenchTracker).

Drives synthetic attempts frame-by-frame: a still, arms-locked setup (so the
judge issues START), a lower-to-chest + hold (PRESS), a press up, then a still,
locked finish (RACK). Bench ignores the depth arg, so a gated placeholder is
passed for each frame.
"""

from __future__ import annotations

import numpy as np

from whitelights.bench import BenchTracker
from whitelights.depth import DepthFrameResult
from whitelights.types import Fault, Verdict

STILL = 25  # frames of holding (0.83s at 30fps) > the 0.6s command holds
TOP = 1.6  # extended-arms wrist height
CHEST = 1.05  # bar-on-chest wrist height


def _ramp(a: float, b: float, n: int = 15) -> list[float]:
    return [float(z) for z in np.linspace(a, b, n)]


def _gated(seq):
    return [
        DepthFrameResult(frame_idx=f.frame_idx, time_s=f.time_s, gated=True) for f in seq.frames
    ]


def _drive(tracker, poses):
    return [tracker.update(f, d) for f, d in zip(poses.frames, _gated(poses), strict=True)]


def _commands(statuses):
    return [s.command for s in statuses if s.command]


def test_good_bench_issues_start_press_rack(make_bench_from_series) -> None:
    series = [TOP] * STILL + _ramp(TOP, CHEST) + [CHEST] * STILL + _ramp(CHEST, TOP) + [TOP] * STILL
    statuses = _drive(BenchTracker(), make_bench_from_series(series))

    assert _commands(statuses) == ["START", "PRESS", "RACK"]
    v = statuses[-1].last_verdict
    assert v.verdict == Verdict.GOOD
    assert v.faults == []


def test_bar_not_to_chest(make_bench_from_series) -> None:
    # Lower only partway (never to the chest), then press.
    series = [TOP] * STILL + _ramp(TOP, 1.35) + _ramp(1.35, TOP) + [TOP] * STILL
    statuses = _drive(BenchTracker(), make_bench_from_series(series))

    assert "START" in _commands(statuses)
    v = statuses[-1].last_verdict
    assert Fault.BAR_NOT_TO_CHEST in v.faults
    assert v.verdict == Verdict.NO_LIFT


def test_pressing_before_press_command(make_bench_from_series) -> None:
    # Touch the chest but press straight back up with no pause -> EARLY_PRESS.
    series = [TOP] * STILL + _ramp(TOP, CHEST) + _ramp(CHEST, TOP) + [TOP] * STILL
    statuses = _drive(BenchTracker(), make_bench_from_series(series))

    v = statuses[-1].last_verdict
    assert Fault.EARLY_PRESS in v.faults
    assert Fault.BAR_NOT_TO_CHEST not in v.faults  # it did reach the chest
    assert v.verdict == Verdict.NO_LIFT


def test_downward_movement_during_press(make_bench_from_series) -> None:
    series = (
        [TOP] * STILL
        + _ramp(TOP, CHEST)
        + [CHEST] * STILL
        + _ramp(CHEST, 1.4, 10)
        + _ramp(1.4, 1.2, 6)  # dip on the way up
        + _ramp(1.2, TOP, 10)
        + [TOP] * STILL
    )
    statuses = _drive(BenchTracker(), make_bench_from_series(series))

    v = statuses[-1].last_verdict
    assert Fault.DOWNWARD_MOVEMENT in v.faults
    assert v.verdict == Verdict.NO_LIFT
