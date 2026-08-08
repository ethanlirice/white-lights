"""Free-rep (training) judges for bench and deadlift.

Drives multi-rep sets frame by frame. The two lifts move in opposite directions —
the bench starts at lockout and goes down, the deadlift starts on the floor and
goes up — so these tests exist partly to prove the shared travel-from-rest
tracker handles both without special-casing either.
"""

from __future__ import annotations

import numpy as np
import pytest
from conftest import make_bench_3d, make_deadlift_3d

from whitelights.depth import DepthFrameResult
from whitelights.freereps import FreeRepState, bench_rep_tracker, deadlift_rep_tracker
from whitelights.judges import supports_training, tracker_for
from whitelights.types import Fault, Verdict

REST = 20  # frames held at rest between reps (0.67 s at 30 fps)

BENCH_TOP, BENCH_CHEST = 1.6, 1.05
DL_FLOOR, DL_TOP = 0.15, 1.0


def _ramp(a: float, b: float, n: int = 12) -> list[float]:
    return [float(z) for z in np.linspace(a, b, n)]


def _cycle(rest: float, extreme: float, *, hold: int = 3) -> list[float]:
    """One out-and-back rep, ending back at rest."""
    return _ramp(rest, extreme) + [extreme] * hold + _ramp(extreme, rest)


def _drive(tracker, poses):
    depths = [
        DepthFrameResult(frame_idx=f.frame_idx, time_s=f.time_s, gated=True) for f in poses.frames
    ]
    return [tracker.update(f, d) for f, d in zip(poses.frames, depths, strict=True)]


def _verdicts(statuses):
    return [s.last_verdict for s in statuses if s.rep_completed]


# ---------------------------------------------------------------------------
# Bench — starts at lockout, travels down
# ---------------------------------------------------------------------------


def test_bench_counts_three_good_reps() -> None:
    series = [BENCH_TOP] * REST
    for _ in range(3):
        series += _cycle(BENCH_TOP, BENCH_CHEST) + [BENCH_TOP] * REST
    statuses = _drive(bench_rep_tracker(), make_bench_3d(series))

    verdicts = _verdicts(statuses)
    assert len(verdicts) == 3, f"expected 3 reps, got {len(verdicts)}"
    assert all(v.verdict == Verdict.GOOD for v in verdicts)
    assert all(v.faults == [] for v in verdicts)
    assert statuses[-1].rep_count == 3


def test_bench_short_rep_is_no_lift_not_uncounted() -> None:
    """Bar stops well above the chest: a counted rep, judged BAR_NOT_TO_CHEST."""
    series = [BENCH_TOP] * REST + _cycle(BENCH_TOP, 1.28) + [BENCH_TOP] * REST
    statuses = _drive(bench_rep_tracker(), make_bench_3d(series))

    verdicts = _verdicts(statuses)
    assert len(verdicts) == 1
    assert verdicts[0].verdict == Verdict.NO_LIFT
    assert Fault.BAR_NOT_TO_CHEST in verdicts[0].faults


def test_bench_twitch_is_not_counted() -> None:
    """A small bob never travels far enough to be a rep."""
    series = [BENCH_TOP] * REST + _cycle(BENCH_TOP, 1.54) + [BENCH_TOP] * REST
    statuses = _drive(bench_rep_tracker(), make_bench_3d(series))

    assert _verdicts(statuses) == []
    assert statuses[-1].rep_count == 0


def test_bench_reversal_during_press_flags_downward_movement() -> None:
    series = (
        [BENCH_TOP] * REST
        + _ramp(BENCH_TOP, BENCH_CHEST)
        + _ramp(BENCH_CHEST, 1.35, 8)  # pressing up
        + _ramp(1.35, 1.18, 6)  # sinks back down
        + _ramp(1.18, BENCH_TOP, 10)
        + [BENCH_TOP] * REST
    )
    statuses = _drive(bench_rep_tracker(), make_bench_3d(series))

    verdicts = _verdicts(statuses)
    assert len(verdicts) == 1
    assert Fault.DOWNWARD_MOVEMENT in verdicts[0].faults


# ---------------------------------------------------------------------------
# Deadlift — starts on the floor, travels up
# ---------------------------------------------------------------------------


def test_deadlift_counts_two_good_reps() -> None:
    series = [DL_FLOOR] * REST
    for _ in range(2):
        series += _cycle(DL_FLOOR, DL_TOP) + [DL_FLOOR] * REST
    statuses = _drive(deadlift_rep_tracker(), make_deadlift_3d(series))

    verdicts = _verdicts(statuses)
    assert len(verdicts) == 2, f"expected 2 reps, got {len(verdicts)}"
    assert all(v.verdict == Verdict.GOOD for v in verdicts)


def test_deadlift_never_locking_out_is_incomplete() -> None:
    series = [DL_FLOOR] * REST + _cycle(DL_FLOOR, 0.72) + [DL_FLOOR] * REST
    statuses = _drive(deadlift_rep_tracker(), make_deadlift_3d(series))

    verdicts = _verdicts(statuses)
    assert len(verdicts) == 1
    assert verdicts[0].verdict == Verdict.NO_LIFT
    assert Fault.INCOMPLETE_LOCKOUT in verdicts[0].faults


def test_deadlift_states_use_ui_recognised_names() -> None:
    """web/live.html pulses the lamp on these exact state strings."""
    series = [DL_FLOOR] * REST + _cycle(DL_FLOOR, DL_TOP) + [DL_FLOOR] * REST
    seen = {s.state for s in _drive(deadlift_rep_tracker(), make_deadlift_3d(series))}

    assert FreeRepState.PULLING in seen
    assert FreeRepState.LOWERING in seen
    assert FreeRepState.READY in seen


# ---------------------------------------------------------------------------
# Shared behaviour and wiring
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("make_tracker", "poses_for"),
    [
        (bench_rep_tracker, lambda s: make_bench_3d(s)),
        (deadlift_rep_tracker, lambda s: make_deadlift_3d(s)),
    ],
    ids=["bench", "deadlift"],
)
def test_free_reps_never_issue_commands(make_tracker, poses_for) -> None:
    """Training mode is uncommanded — that is the whole distinction."""
    rest, extreme = (
        (BENCH_TOP, BENCH_CHEST)
        if make_tracker is bench_rep_tracker
        else (
            DL_FLOOR,
            DL_TOP,
        )
    )
    series = [rest] * REST + _cycle(rest, extreme) + [rest] * REST
    statuses = _drive(make_tracker(), poses_for(series))

    assert all(s.command is None for s in statuses)


def test_holding_still_produces_no_reps() -> None:
    statuses = _drive(bench_rep_tracker(), make_bench_3d([BENCH_TOP] * 60))
    assert _verdicts(statuses) == []


def test_every_lift_now_offers_training() -> None:
    for lift in ("squat", "bench", "deadlift"):
        assert supports_training(lift) is True


def test_training_returns_a_free_rep_judge_for_every_lift() -> None:
    """The regression that started this: training must not hand back a
    single-attempt competition judge."""
    for lift in ("bench", "deadlift"):
        tracker = tracker_for(lift, "training")
        assert type(tracker).__name__ == "FreeRepTracker"
        competition = tracker_for(lift, "competition")
        assert type(competition).__name__ != "FreeRepTracker"
